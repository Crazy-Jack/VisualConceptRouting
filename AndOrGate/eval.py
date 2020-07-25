"""Evaluation of the model"""

import os
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
from PIL import Image

from data_utlis import LsunBedDataset
from data_utlis import MyTransform
from network import Critic, Generator, Encoder, Models
from utlis import save_model
from utlis import txt_logger

def set_args():
    parser = argparse.ArgumentParser("Visual Concept routing")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count()-3,
                        help='num of workers to use when load data')
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

    parser.add_argument('--save_folder', type=str, default='../train_related/testing')


    parser.add_argument('--generator_model_path', type=str, required=True, help="generator model path")
    parser.add_argument('--encoder_model_path', type=str, required=True, help="encoder model path")

    parser.add_argument('--z_dim', type=int, default=100, help="hidden vector dim")


    args = parser.parse_args()

    args.data_root_folder = os.path.join(args.data_folder, args.data_root_name)
    return args


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

def load_model(args, model, mode):
    if mode == 'generator':
        model_path = args.generator_model_path
    elif mode == 'encoder':
        model_path = args.encoder_model_path
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model

def save_batch_img(args, data_loader, generator, encoder):

    save_path_recon = os.path.join(args.save_folder, 'reconstruct')
    save_path_generation = os.path.join(args.save_folder, 'generation')
    os.makedirs(save_path_generation, exist_ok=True)
    # reconstruction
    for batch_id, img in tqdm(enumerate(data_loader)):
        bsz = img.shape[0]
        with torch.no_grad():
            latent_z = encoder(img)
            recon_img_batch = generator(latent_z) # bsz, c, h, w
        for index in range(bsz):
            origin_img = img_from_batch(img[index]).convert('RGB')
            recon_img = img_from_batch(recon_img_batch[index]).convert('RGB')
            subfolder_name = 'batch_{}_index_{}'.format(batch_id, index)
            img_save_folder = os.path.join(save_path_recon, subfolder_name)
            os.makedirs(img_save_folder, exist_ok=True)
            origin_img.save(os.path.join(img_save_folder, 'origin.png'))
            recon_img.save(os.path.join(img_save_folder, 'reconstruct.png'))
        break
    # generation
    sample_z = np.random.uniform(-1, 1, (args.batch_size, args.z_dim)) # their code exactly
    sample_z = torch.FloatTensor(sample_z)
    generate_img_batch = generator(sample_z).detach()
    for index in range(generate_img_batch.shape[0]):
        gene_img = img_from_batch(generate_img_batch[index]).convert('RGB')
        gene_img.save(os.path.join(save_path_generation, '{}.png'.format(index)))
    
        
def img_from_batch(input_tensor):
    img = Image.fromarray(np.uint8(input_tensor.numpy()*255))
    return img





def main():
    args = set_args()
    encoder = Encoder(out_dim=args.z_dim)
    generator = Generator(z_dim=args.z_dim)

    encoder = load_model(args, encoder, 'encoder')
    generator = load_model(args, generator, 'generator')

    loader = set_loader(args)
    save_batch_img(args, loader, generator, encoder)


if __name__ == "__main__":
    main()
