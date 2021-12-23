# %%
'''
pure dcgan structure.
code similar sample from the pytorch code.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn

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

        self.output_imgsize = (self.channels, self.imsize, self.imsize)
        
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.n_classes + self.z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.output_imgsize))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # concatenate label embedding and image to produce input 
        gen_input = torch.cat((self.label_emb(labels), z), -1)    
        img = self.model(gen_input)    
        img = img.view(img.size(0), *self.output_imgsize)

        return img

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
        
        self.model = nn.Sequential(
            nn.Linear((self.n_classes + int(np.prod(self.input_imsize))), 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, imgs, labels):
        # concatenate label embedding and image to produce input 
        d_in = torch.cat((imgs.view(imgs.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)

        return validity