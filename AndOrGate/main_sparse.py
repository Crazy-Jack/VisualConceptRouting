"""
Pytorch Implementation for Inducing Hierarchical Compositional Model by Sparsifying Generator Network. CVPR 2020.
"""
import os, sys
import argparse
import numpy as np
import shutil
import multiprocessing


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import tensorboard_logger as tb_logger
import pandas as pd
from torch.autograd import grad as torch_grad

from data_utlis import LsunBedDataset
from data_utlis import MyTransform
from network import Critic, Generator, Encoder, Models
from utlis import save_model
from utlis import txt_logger


def set_args():
    parser = argparse.ArgumentParser("Visual Concept routing")
    # data
    parser.add_argument('--dataset', type=str, default= 'lsun-bed',
                        choices=['lsun-bed'], help="choose which dataset are using")
    parser.add_argument('--data_folder', type=str, default='../data_unzip/bedroom_train_lmdb/lsun_bed100k', help='dataset')
    parser.add_argument('--data_root_name', type=str, default='imgs', choices=['imgs'],
                        help="dataset img folder name, only needed when dataset is organized by folders of img")
    parser.add_argument('--meta_file_train', type=str, default='meta_lsun_100k.csv',
                        help='meta data for ssl training')
    parser.add_argument('--img_size', type=int, default=64,
                        help='img size used in training')
    # train
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count()-3,
                        help='num of workers to use when load data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--parallel', action='store_true', help="true if using data parallel")
    parser.add_argument('--save_freq', type=int, default=10, help="save freq for models")

    # loss weight
    parser.add_argument('--recon_weight', type=float, default=0.1,
                        help='reconstruction weight')
    parser.add_argument('--real_critic_weight', type=float, default=1e-3,
                        help='weight in critic real score')
    parser.add_argument('--gp_weight', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='dimemtion for latent z')

    # optimizer
    parser.add_argument('--lr_encoder', type=float, default=7e-4,
                        help='lr for encoder')
    parser.add_argument('--lr_generator', type=float, default=7e-4,
                    help='lr for generator')
    parser.add_argument('--lr_critic', type=float, default=7e-4,
                    help='lr for critic')
    # other
    parser.add_argument('--resume_model_path', type=str, default='0',
                    help='from with model training would resume')


    args = parser.parse_args()

    args.data_root_folder = os.path.join(args.data_folder, args.data_root_name)
    args.model_path = '../train_related/VisulConceptRouting/SparseVAEGAN/{}_models_'.format(args.dataset)
    args.tb_path = '../train_related/VisulConceptRouting/SparseVAEGAN/{}_tensorboard_'.format(args.dataset)

    if args.resume_model_path != '0':
        args.pre_ssl_epoch = int(opt.resume_model_path.split('/')[-1].split('.')[0].split('_')[-1])
        args.model_path += '_resume_from_epoch_{}'.format(args.pre_ssl_epoch)
        args.tb_path += '_resume_from_epoch_{}'.format(args.pre_ssl_epoch)


    args.model_name = '{}_recon_weight_{}_zdim_{}_lrencode_{}_lrgene_{}_lrcritic_{}'.\
        format(args.dataset, args.recon_weight, args.z_dim, args.lr_encoder, args.lr_generator, args.lr_critic)

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if os.path.isdir(args.tb_folder):
        delete = input("Are you sure to delete folder {}？ (Y/n)".format(args.tb_folder))
        if delete.lower() == 'y':
            shutil.rmtree(args.tb_folder)
        else:
            sys.exit("{} FOLDER is untorched.".format(args.tb_folder))
    os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def gradient_penalty(real_data, generated_data, Critic, args):
    batch_size = real_data.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand_as(real_data)
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
    return args.gp_weight * ((gradients_norm - 1) ** 2).mean()


def set_loader(args):
    """Configure dataloader"""
    train_transform = MyTransform(args).train_transform()

    if args.dataset == 'lsun-bed':
        train_df = pd.read_csv(os.path.join(args.data_folder, args.meta_file_train))
        train_dataset = LsunBedDataset(train_df, root=args.data_root_folder,
                                    transform=train_transform)
    else:
        raise ValueError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader

def set_model(args):
    """Configure models and loss"""
    # models
    encoder = Encoder(out_dim=args.z_dim).cuda()
    generator = Generator(z_dim=args.z_dim).cuda()
    critic = Critic().cuda()

    if torch.cuda.device_count() > 1:
        print("Use device count: {}".format(torch.cuda.device_count()))
        encoder = torch.nn.DataParallel(encoder)
        generator = torch.nn.DataParallel(generator)
        critic = torch.nn.DataParallel(critic)
        encoder.cuda()
        generator.cuda()
        critic.cuda()
        cudnn.benchmark = True

    models = Models(encoder, generator, critic)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    optim_generator = torch.optim.Adam(generator.parameters(), lr=args.lr_generator)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # critieron
    l2_reconstruct_criterion = nn.MSELoss().cuda()

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
        optim_encoder.zero_grad()
        optim_generator.zero_grad()
        loss.backward()
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
            latent_z = Variable(torch.FloatTensor(sample_z), requires_grad=False).cuda()
            generated_img = models.generator(latent_z).detach()

        fake_score = models.critic(generated_img)
        real_score = models.critic(img)
        # wGAN loss
        loss_1 = fake_score.mean() - real_score.mean()
        # gradient penalty
        loss_2 = gradient_penalty(img, generated_img, models.critic, args)
        # real critic constrain
        loss_3 = args.real_critic_weight * (torch.square(real_score)).mean()

        loss = loss_1 + loss_2 + loss_3
        losses += loss.item()

        optim_critic.zero_grad()
        loss.backward()
        optim_critic.step()

    losses = losses / num_data

    return losses


def main():
    args = set_args()
    # tensorboard and logger
    tf_logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    scalar_logger = txt_logger(args.save_folder, args)

    # set loader
    train_loader = set_loader(args)

    # get model and optimizers
    models, optim_encoder, optim_generator, optim_critic, l2_reconstruct_criterion = set_model(args)

    # train routine
    for epoch in range(1, args.epochs + 1):
        # train encoder and generator
        recon_losses, critic_losses = train_encoder_generator(train_loader, models, optim_encoder, optim_generator, l2_reconstruct_criterion, args)

        # train critic
        train_critic_loss = train_critic(train_loader, models, optim_critic, args)

        # logging
        tf_logger.log_value('recon_losses', recon_losses, epoch)
        tf_logger.log_value('critic_losses', critic_losses, epoch)
        tf_logger.log_value('train_critic_loss', train_critic_loss, epoch)
        scalar_logger.log_value(epoch, ('recon_losses', recon_losses),
                                       ('critic_losses', critic_losses),
                                       ('train_critic_loss', train_critic_loss))

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_model(models.encoder, optim_encoder, args, epoch, os.path.join(args.save_folder, 'ckpt_encoder_epoch_{}.ckpt'.format(epoch)))
            save_model(models.generator, optim_generator, args, epoch, os.path.join(args.save_folder, 'ckpt_generator_epoch_{}.ckpt'.format(epoch)))
            save_model(models.critic, optim_critic, args, epoch, os.path.join(args.save_folder, 'ckpt_critic_epoch_{}.ckpt'.format(epoch)))

if __name__ == "__main__":
    """Command
    $ python main_sparse.py
    """
    main()
