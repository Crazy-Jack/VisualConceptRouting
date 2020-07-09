import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


#######################################################
#                   Down Sample Block                 #
#######################################################
class BasicBlockDown(nn.Sequential):
    def __init__(self, in_dim, hidden_dim1):
        super(BasicBlockDown, self).__init__(
            nn.Conv2d(in_dim, hidden_dim1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim1, hidden_dim1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2) # 32x32
        )

class ResBlockDown(nn.Module):
    def __init__(self, in_dim, hidden_dim1):
        super(ResBlockDown, self).__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_dim, hidden_dim1, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, hidden_dim1, 3, 1, 1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(hidden_dim1, hidden_dim1, 3, 1, 1)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
    
    def forward(self, x):
        shortcut = self.conv1(self.avgpool1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.avgpool2(x)
        return x + shortcut

#######################################################
#                   Up Sample Block                   #
#######################################################    
class BasicBlockUp(nn.Sequential):
    def __init__(self, in_dim, hidden_dim1, scale_factor=2):
        super(BasicBlockUp, self).__init__(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_dim, hidden_dim1, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim1, hidden_dim1, 3, 1, 1),
            nn.ReLU(),
        )


class ResBlockUp(nn.Module):
    """Residual Block for upsampling"""
    def __init__(self, in_dim, hidden_dim1, scale_factor=2):
        super(ResBlockUp, self).__init__()
        self.conv2d_shortcut = nn.Conv2d(in_dim, hidden_dim1, 3, 1, 1)
        self.upsample_shortcut = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv2d1 = nn.Conv2d(in_dim, hidden_dim1, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(hidden_dim1, hidden_dim1, 3, 1, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        shortcut = self.upsample_shortcut(self.conv2d_shortcut(x))
        x = self.upsample1(x)
        x = self.relu1(self.conv2d1(x))
        x = self.relu2(self.conv2d2(x))

        return x + shortcut

#######################################################
#              Encoder, Generator, Critic             #
#######################################################       
class Encoder(nn.Module):
    """Encoder for embedding img.
    >>> encoder = Encoder(out_dim=123)
    >>> img_batch = torch.rand([10, 3, 64, 64])
    >>> feature = encoder(img_batch)
    >>> print(feature.shape)
    torch.Size([10, 123])
    """
    def __init__(self, out_dim):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 5, 1, 2) # [bsz, 3, 64, 64]
        self.lrelu0 = nn.LeakyReLU(0.2)
        self.block1 = BasicBlockDown(64, 64) # [bsz, 64, 32, 32]
        self.block2 = ResBlockDown(64, 64 * 2) # [bsz, 64 * 2, 16, 16]
        self.block3 = BasicBlockDown(64 * 2, 64 * 4) # [bsz, 64 * 4, 8, 8]
        self.block4 = BasicBlockDown(64 * 4, 64 * 8) # [bsz, 64 * 8, 4, 4]
        self.flat = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * (64 * 8), out_dim) # [bsz, out_dim]
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.lrelu0(self.conv0(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.tanh(self.fc(self.flat(x)))

        return x


class Critic(nn.Sequential):
    """Critic for WGAN/Energy-based model
    >>> critic = Critic()
    >>> img_batch = torch.rand([10, 3, 64, 64])
    >>> feature = critic(img_batch)
    >>> print(feature.shape)
    torch.Size([10, 1])
    """
    def __init__(self, out_dim=1):
        super(Critic, self).__init__(
            nn.Conv2d(3, 64, 5, 1, 2), # [bsz, 3, 64, 64]
            nn.LeakyReLU(0.2),
            BasicBlockDown(64, 64), # [bsz, 64, 32, 32]
            ResBlockDown(64, 64 * 2), # [bsz, 64 * 2, 16, 16]
            BasicBlockDown(64 * 2, 64 * 4), # [bsz, 64 * 4, 8, 8]
            BasicBlockDown(64 * 4, 64 * 8), # [bsz, 64 * 8, 4, 4]
            nn.Conv2d(64 * 8, 64 * 8, 3, 1, 1), # [bsz, 64 * 8, 4, 4]
            nn.LeakyReLU(0.2),
            nn.Flatten(), # [bsz, 4 * 4 * (64 * 8)]
            nn.Linear(4 * 4 * (64 * 8), out_dim), # [bsz, out_dim]
        )


class Generator(nn.Module):
    """Sparse Generator
    >>> generator = Generator(100)
    >>> z_batch = torch.rand([10, 100])
    >>> img = generator(z_batch)
    >>> print(img.shape)
    torch.Size([10, 3, 64, 64])
    """
    def __init__(self, z_dim, hidden_dim=64, out_channel=3):
        super(Generator, self).__init__()
        self.fc0 = nn.Linear(z_dim, 4 * 4 * (hidden_dim * 8)) # [bsz, 4 * 4 * (64*8)] -> reshape to [bsz, 64 * 8, 4, 4]
        self.relu0 = nn.ReLU()
        self.conv2d1 = nn.Conv2d(hidden_dim * 8, 200, 3, 1, 1) # [bsz, 200, 4, 4]
        self.relu1 = nn.ReLU()
        self.sparse2 = nn.Conv2d(200, 200 * 4 * 4, 3, 1, 1) # TODO: k=? Fill sparsity code: change from [bsz, 200, 4, 4] -> [bsz, 200 * 4 * 4, 4, 4]
        self.block3 = BasicBlockUp(200 * 4 * 4, hidden_dim * 8) # [bsz, 64 * 8, 8, 8]
        self.sparse3 = Sparsify_hw(topk= 8 * 8 / 4) # TODO: change sparse operation; k = 8 * 8 / 4
        self.block4 = BasicBlockUp(hidden_dim * 8, hidden_dim * 4) # [bsz, 64 * 4, 16, 16]
        self.sparse4 = Sparsify_hw(topk= 16 * 16 / 4) # TODO: change sparse operation; k = 16 * 16 / 4
        self.block5 = ResBlockUp(hidden_dim * 4, hidden_dim * 2) # [bsz, 64 * 2, 32, 32]
        self.sparse5 = Sparsify_hw(topk = 32 * 32 / 4) # TODO: change sparse operation; k = 32 * 32 / 4
        self.block6 = BasicBlockUp(hidden_dim * 2, hidden_dim) # [bsz, 64, 64, 64]
        self.conv2d7 = nn.Conv2d(hidden_dim, out_channel, 5, 1, 2) # [bsz, 3, 64, 64]
        self.tanh7 = nn.Tanh()

    def forward(self, x):
        x = self.relu0(self.fc0(x))
        x = self.relu1(self.conv2d1(x.reshape([-1, 64 * 8, 4, 4])))
        x = self.sparse2(x) # [bsz, 200 * 4 * 4, 4, 4]
        x = self.sparse3(self.block3(x)) # [bsz, 64 * 8, 8, 8]
        x = self.sparse4(self.block4(x)) # [bsz, 64 * 4, 16, 16]
        x = self.sparse5(self.block5(x)) # [bsz, 64 * 2, 32, 32]
        x = self.block6(x) # [bsz, 64, 64, 64]
        x = self.tanh7(self.conv2d7(x)) # [bsz, 3, 64, 64]

        return x

class Sparsify(nn.Module):
    """Sparsify tensors on specific dim
    """
    def __init__(self, topk):
        super(Sparsify, self).__init__()
        self.topk = topk
    
    def forward(self, x, sparse_dim=1): 
        # Input x is [bsz, channel, imgs, imgs]

        assert self.topk <= x.shape[sparse_dim], "Sparse K ({}) is larger or equal to the sparse dim ({})".format(self.topk, x.shape[sparse_dim])
        _, index = torch.topk(x, self.topk, dim=sparse_dim)
        mask = torch.zeros_like(x.shape).scatter_(sparse_dim, index, 1)
        sparsed_x = mask * x
        return sparsed_x

class Sparsify_hw(nn.Module):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
    def forward(self, x):
        n,c,h,w = x.shape
        x_reshape = x.view(n,c,h*w)
        _, index = torch.topk(x, self.topk, dim=2)
        mask = torch.zeros_like(x_reshape).scatter_(dim=2, index=index, src=1.0)
        sparse_x = mask * x_reshape
        return sparse_x.view(n,c,h,w)

class Models:
    def __init__(self, encoder, generator, critic):
        self.encoder = encoder
        self.generator = generator
        self.critic = critic
    def train(self):
        self.encoder.train()
        self.generator.train()
        self.critic.train()

    def eval(self):
        self.encoder.eval()
        self.generator.eval()
        self.critic.eval()



if __name__ == "__main__":
    import doctest
    doctest.testmod()



# class Generator(nn.Module):
#     def __init__(self, z_dim, hidden_dim=64, out_channel=3):
#         super().__init__()

#         # Input is the latent vector Z.
#         self.tconv1 = nn.ConvTranspose2d(z_dim, hidden_dim * 8,
#             kernel_size=4, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_dim * 8)

#         # Input Dimension: (hidden_dim*8) x 4 x 4
#         self.tconv2 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4,
#             4, 2, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(hidden_dim * 4)

#         # Input Dimension: (hidden_dim*4) x 8 x 8
#         self.tconv3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2,
#             4, 2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(hidden_dim * 2)

#         # Input Dimension: (hidden_dim*2) x 16 x 16
#         self.tconv4 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim,
#             4, 2, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(hidden_dim)

#         # Input Dimension: (hidden_dim) * 32 * 32
#         self.tconv5 = nn.ConvTranspose2d(hidden_dim, 3,
#             4, 2, 1, bias=False)
#         # Output Dimension: (out_channel) x 64 x 64

#     def forward(self, x):
#         x = F.relu(self.bn1(self.tconv1(x)))
#         x = F.relu(self.bn2(self.tconv2(x)))
#         x = F.relu(self.bn3(self.tconv3(x)))
#         x = F.relu(self.bn4(self.tconv4(x)))
#         x = F.tanh(self.tconv5(x))
#         return x
