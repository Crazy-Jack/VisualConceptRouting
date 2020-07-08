"""
Pytorch Implementation for Inducing Hierarchical Compositional Model by Sparsifying Generator Network. CVPR 2020.
"""
import os
import argparse
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets

from data_utlis import LsunDataset
from data_utlis import MyTransform
from network import Critic, Generator, Encoder, Models
from utlis import dotdict
from torch.autograd import Variable

def set_args():
    parser = argparse.ArgumentParser("Visual Concept routing")
    # data
    parser.add_argument('--dataset', type=str, default= 'lsun-bed',
                        choices=['lsun-bed'], help="choose which dataset are using")
    parser.add_argument('--data_folder', type=str, default='../data_unzip/bedroom_train_lmdb', help='dataset')
    parser.add_argument('--data_root_name', type=str, default='imgs', choices=['imgs'],
                        help="dataset img folder name, only needed when dataset is organized by folders of img")
    parser.add_argument('--meta_file_train', type=str, default='meta_lsun_100k.csv',
                        help='meta data for ssl training')
    # train
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use when load data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    
    # generator
    parser.add_argument('--recon_weight', type=float, default=0.1,
                        help='reconstruction weight')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='dimemtion for latent z')
    real_norm_weight
    
    
    args = parser.parse_args()


    return args


def gradient_penalty(real_data, generated_data, Critic, device, args):
    batch_size = real_data.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()
    # Calculate probability of interpolated examples
    prob_interpolated = Critic(interpolated) 

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return args.gpweight * ((gradients_norm - 1) ** 2).mean()


def set_loader(args):
    """Configure dataloader"""
    train_transform = MyTransform(args).train_transform()

    if args.dataset == 'lsun-bed':
        train_df = pd.read_csv(os.path.join(args.data_folder, args.meta_file_train))
        train_dataset = LsunDataset(train_df, root=os.path.join(args.data_folder, args.data_root_name),
                                    transform=train_transform)
    else:
        raise ValueError(args.datasets)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader

def set_model(args):
    """Configure models and loss"""

    # models
    encoder = Encoder()
    generator = Generator()
    critic = Critic()

    models = Models(encoder, generator, critic)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    optim_generator = torch.optim.Adam(generator.parameters(), lr=args.lr_generator)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_generator)

    # critieron
    l2_reconstruct_criterion = nn.MSELoss()

    return models, optim_encoder, optim_generator, optim_critic, l2_reconstruct_criterion


def train_encoder_generator(train_loader, models, optim_encoder, optim_generator, l2_reconstruct_criterion, args):
    """Train for one epoch"""
    models.train()
    recon_losses = 0.0
    critic_losses = 0.0
    num_data = 0.0
    for batch_id, img in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = img.cuda()
        num_data = img.shape[0]
        # encoder img to reconstruct
        z_a = models.encoder(img)
        reconstruct_img = models.generator(z_a)
        reconstruct_loss = l2_reconstruct_criterion(reconstruct_img, img)
        # feed in critic
        recon_score = models.critic(reconstruct_img)
        critic_loss = - recon_score.mean()
        # final loss
        loss = reconstruct_loss * args.recon_weight + critic_loss
        # update
        loss.backward()
        optim_encoder.zero_grad()
        optim_generator.zero_grad()
        optim_encoder.step()
        optim_generator.step()
        
        recon_losses += reconstruct_loss.item()
        critic_losses += critic_loss.item()
    recon_losses = recon_losses / num_data
    critic_losses = critic_losses / num_data
    return recon_losses, critic_losses
    

def train_critic(train_loader, models, optim_critic, args):
    models.train()
    num_data = 0.0
    losses = 0.0
    for batch_id, img in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = img.cuda()
        bzs = img.shape[0]
        num_data += bzs
        with torch.no_grad():
            sample_z = np.random.uniform(-1, 1, (bzs, args.z_dim)) # their code exactly
            latent_z = Variable(torch.FloatTensor(sample_z), requires_grad=False)
            generated_img = models.generator(latent_z).detach()
        
        fake_score = models.critic(generated_img)
        real_score = models.critic(img)
        # wGAN loss
        loss_1 = fake_score.mean() - real_score.mean()
        # gradient penalty
        loss_2 = gradient_penalty(img, generated_img, Critic, args)
        # real critic constrain
        loss_3 = args.real_critic_weight * (torch.square(real_score)).mean()

        loss = loss1 + loss_2 + loss_3
        losses += loss.item()
        loss.backward()
        optim_critic.zero_grad()
        optim_critic.step()
    losses = losses / num_data
    return losses


def main():
    args = set_args()

    train_loader = set_loader(args)
    for epoch in args.epochs:
        models, optim_encoder, optim_generator, optim_critic, l2_reconstruct_criterion = set_model()
        recon_losses, critic_losses = train_encoder_generator(train_loader, models, optim_encoder, optim_generator, l2_reconstruct_criterion, args)
        critic_loss = train_critic(train_loader, models, optim_critic, args)




if __name__ == "__main__":
    main()