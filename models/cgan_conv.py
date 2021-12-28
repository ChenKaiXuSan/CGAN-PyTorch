# %%
'''
conv network condtional gan structure.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# %%
class Generator(nn.Module):
    '''
    the conditional GAN generator structure from paper.

    '''    
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels = 1, n_classes = 10):
        
        super(Generator, self).__init__()

        self.imsize = image_size
        self.channels = channels
        self.z_dim = z_dim
        self.n_classes = n_classes
        
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            nn.ConvTranspose2d(self.n_classes + self.z_dim, conv_dim * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # concatenate label embedding and image to produce input 
        gen_input = torch.cat((self.label_emb(labels), z), -1)    
        gen_input_flat = gen_input.unsqueeze(2).unsqueeze(3)

        out = self.l1(gen_input_flat)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = self.last(out)

        return out

# %%
class Discriminator(nn.Module):
    '''
    the conditional GAN discriminator structure from paper.

    '''
    def __init__(self, image_size = 64, conv_dim = 64, channels = 1, n_classes = 10):

        super(Discriminator, self).__init__()

        self.imsize = image_size
        self.channels = channels
        self.n_classes = n_classes
        self.input_imsize = (self.channels, self.imsize, self.imsize)

        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)
        
        # (*, 1, 64, 64)
        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels + self.n_classes, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim
        # (*, 64, 32, 32)
        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        # (*, 128, 16, 16)
        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        curr_dim = curr_dim * 2
        # (*, 256, 8, 8)
        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        
        # output layers
        # (*, 512, 4, 4)
        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
            # without sigmoid, used in the loss funciton
            )

    def forward(self, imgs, labels):
        # labels_one_hot = F.one_hot(labels, num_classes = self.n_classes)

        labels_embedding = self.label_embedding(labels)
        labels_flat = labels_embedding.unsqueeze(2).unsqueeze(3).expand(imgs.size(0), labels_embedding.size(1), self.imsize, self.imsize)
        # (*, 10, 64, 64)
        # concatenate label embedding and image to produce input 
        d_in = torch.cat((imgs, labels_flat), 1)

        out = self.l1(d_in)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        validity = self.last_adv(out)

        return validity.squeeze()