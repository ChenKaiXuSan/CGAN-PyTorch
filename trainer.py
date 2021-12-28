# %% 
"""
wgan with different loss function, used the pure dcgan structure.
"""
import os 
import time
import torch
import datetime

import torch.nn as nn 
import torchvision

# from models.cgan import Generator, Discriminator
from models.cgan_conv import Generator, Discriminator
from utils.utils import *

# %%
class Trainer_cgan(object):
    def __init__(self, data_loader, config):
        super(Trainer_cgan, self).__init__()

        # data loader 
        self.data_loader = data_loader

        # exact model and loss 
        self.model = config.model

        # model hyper-parameters
        self.imsize = config.img_size 
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.channels = config.channels
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.n_classes = config.n_classes

        self.epochs = config.epochs + 1
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr 
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset 
        self.use_tensorboard = config.use_tensorboard

        # path
        self.image_path = config.dataroot 
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.version = config.version

        # step 
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # path with version
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

        # start with trained model 
        # if self.pretrained_model:
        #     self.load_pretrained_model()

    def train(self):
        '''
        Training
        '''

        # fixed input for debugging
        # I think if the batch size very large, the cpu memory will oversize.
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim)) # (*, 100)
        fixed_labels = tensor2var(torch.randint(0, self.n_classes, (self.batch_size,))) # (*, )

        print('start training! \n')
        
        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            #swithc to the train mode.
            self.D.train()
            self.G.train()

            for i, (real_images, real_labels) in enumerate(self.data_loader):

                # configure input 
                real_images = tensor2var(real_images)
                real_labels = tensor2var(real_labels)
                
                # adversarial ground truths
                real = tensor2var(torch.full((real_images.size(0),), 0.9)) # (*, )
                fake = tensor2var(torch.full((real_images.size(0),), 0.0)) #(*, )
                
                # ==================== Train D ==================
                # set D gradients to zero.
                self.D.zero_grad()

                # compute loss with real images 
                d_out_real = self.D(real_images, real_labels)

                d_loss_real = self.adversarial_loss(d_out_real, real)

                # noise z for generator
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim)) # 64, 100
                fake_labels = tensor2var(torch.randint(0, self.n_classes, (real_images.size(0),)))

                fake_images = self.G(z, fake_labels) # (*, c, 64, 64)
                d_out_fake = self.D(fake_images.detach(), fake_labels) # (*,)

                d_loss_fake = self.adversarial_loss(d_out_fake, fake)

                # total d loss
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                # update D
                self.d_optimizer.step()

                # train the generator every 5 steps
                if i % self.g_num == 0:

                    # =================== Train G and gumbel =====================
                    # set generator gradients to zero.
                    self.G.zero_grad()

                    # create random noise 
                    # fake_images = self.G(z, fake_labels)

                    # compute loss with fake images 
                    g_out_fake = self.D(fake_images, fake_labels) # batch x n

                    g_loss_fake = self.adversarial_loss(g_out_fake, real)

                    g_loss_fake.backward()
                    # update G
                    self.g_optimizer.step()

            # log to the tensorboard
            self.logger.add_scalar('d_loss', d_loss.data, epoch)
            self.logger.add_scalar('g_loss', g_loss_fake.data, epoch)
            # end one epoch

            # print out log info
            if (epoch) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, "
                    .format(elapsed, epoch, self.epochs, epoch,
                            self.epochs, d_loss.item(), g_loss_fake.item()))

            # sample images 
            if (epoch) % self.sample_step == 0:
                self.G.eval()
                # save real image
                save_sample(self.sample_path + '/real_images/', real_images, epoch)
                
                with torch.no_grad():
                    fake_images = self.G(fixed_z, fixed_labels)
                    # save fake image 
                    save_sample(self.sample_path + '/fake_images/', fake_images, epoch)
                    
                # sample sample one images
                save_sample_one_image(self.sample_path, real_images, fake_images, epoch)

            # save model checkpoint
            if (epoch) % self.model_save_step == 0:
                torch.save({
                    'epoch': epoch,
                    'G_state_dict': self.G.state_dict(),
                    # 'G_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'g_loss': g_loss_fake,
                    'D_state_dict': self.D.state_dict(),
                    'd_loss': d_loss,
                },
                os.path.join(self.model_save_path, '{}.pth.tar'.format(epoch))
                )


    def build_model(self):

        self.G = Generator(image_size = self.imsize, z_dim = self.z_dim, conv_dim = self.g_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()
        self.D = Discriminator(image_size = self.imsize, conv_dim = self.d_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()

        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        # self.G.apply(weights_init)
        # self.D.apply(weights_init)
        
        # optimizer 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # for orignal gan loss function
        self.adversarial_loss = nn.MSELoss().cuda()
        # self.adversarial_loss = nn.BCEWithLogitsLoss().cuda()

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()
