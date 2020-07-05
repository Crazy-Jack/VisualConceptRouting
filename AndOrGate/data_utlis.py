import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import pickle
from PIL import Image
import os
from tqdm import tqdm



class MyTransform:
    """Class for costomize transform"""
    def __init__(self, args):
        super(MyTransform).__init__()
        # normolize
        if args.dataset == 'lsun-bed':
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        else:
            raise ValueError('dataset not supported: {}'.format(args.dataset))
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def train_transform(self, tosize=(64,64), interpolation=Image.LANCZOS):
        """Transform for train"""
        train_transform = transforms.Compose([
            transforms.Resize(size=tosize, interpolation=interpolation),
            transforms.ToTensor(),
            self.normalize,
        ])

        return train_transform


class LsunDataset(Dataset):
    """Dataset for LSUN benchmark"""
    def __init__(self, root='../data_unzip/bedroom_train_lmdb/imgs', transform=None):
        super(LsunDataset, self).__init__()
        self.root = root
        self.img_list = os.listdir(self.root)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        self.img_filename = os.path.join(self.root, self.img_list[index])
        img = Image.open(self.img_filename).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, 0

def image_statistics(data_loader):
    means = torch.zeros([3,])
    stds = torch.zeros([3,])

    # mean (utzap): tensor([0.8342, 0.8142, 0.8081]) 
    num_data = 0.0
    for batch_id, (img,_) in tqdm(enumerate(data_loader), total=len(data_loader)):
        means += img.sum((0, 2, 3)) # [3, ]
        #print(means)
        num_data += img.shape[0]
    pixs = img.shape[2] * img.shape[3]
    means = means / (num_data * pixs)
    print("unbias means:", means)

    # stds(utzap): tensor([0.2804, 0.3014, 0.3072])
    # means = torch.FloatTensor([0.8342, 0.8142, 0.8081])
    num_data = 0.0
    means = means.unsqueeze(1).unsqueeze(2).unsqueeze(0) # [1, 3, 1, 1]
    var = torch.zeros([3,])
    for batch_id, (img, _) in tqdm(enumerate(data_loader), total=len(data_loader)): 
        delta = torch.pow(img - means, 2)
        var += delta.sum((0,2,3))
        num_data += img.shape[0]
        #print(var)
    pixs = img.shape[2] * img.shape[3]
    stds = torch.sqrt(var / (num_data * pixs))
    print("stds: ", stds)  

    return means, stds

def main():
    from main_sparse import set_args
    args = set_args()
    train_transform_ = MyTransform(args).train_transform()
    train_dataset = LsunDataset(root='../data_unzip/bedroom_train_lmdb/imgs', transform=train_transform_)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    
    mean, stds = image_statistics(train_loader)


if __name__ == "__main__":
    main()