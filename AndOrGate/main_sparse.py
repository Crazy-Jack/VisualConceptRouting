"""
Pytorch Implementation for Inducing Hierarchical Compositional Model by Sparsifying Generator Network. CVPR 2020.
"""
import os
import argparse
import numpy as np

from tqdm import tqdm
import torch
from torchvision import datasets

from data_utlis import LsunDataset
from data_utlis import MyTransform
from network import Critic, Generator, Encoder

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
    
    args = parser.parse_args()


    return args


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

def set_model():
    """Configure models and loss"""

    # models
    encoder = Encoder()
    generator = Generator()
    critic = Critic()


    pass

def train(train_loader, args):
    """Train for one epoch"""
    for batch_id, (img, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
        pass
        
        






def gradient_penalty(real_data, generated_data, DNet, mask, num_class, device, args, local=True):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = DNet(interpolated, mask) # mask = 1 if local=False

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon

    if local:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, keepdim=True) + 1e-12)
        gradients_norm = gradients_norm * mask
        return args.gpweight * ((gradients_norm - 1) ** 2).mean(dim=0)
    else:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return args.gpweight * ((gradients_norm - 1) ** 2).mean()




def main():
    args = set_args()

    train_loader = set_loader(args)

    img, label = train(train_loader, args)
    print(img.shape)
    print(label.shape)


if __name__ == "__main__":
    main()