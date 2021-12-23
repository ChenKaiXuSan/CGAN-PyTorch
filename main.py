# %%
import os
import argparse
import random
import pprint
from torch._C import set_anomaly_enabled

import torch.backends.cudnn as cudnn

from trainer import Trainer_cgan
from utils.utils import *
from dataset.dataset import getdDataset

# set the gpu number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='cgan', choices=['gan', 'cgan'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--g_num', type=int, default=5, help='train the generator every 5 steps')
    parser.add_argument('--z_dim', type=int, default=100, help='noise dim')
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--n_classes', type=int, default=10, help='the class number of the dataset')
    parser.add_argument('--version', type=str, default='test', help='the version of the path, for implement')

    # Training setting
    parser.add_argument('--epochs', type=int, default=10000, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2)

    # TTUR 
    parser.add_argument('--g_lr', type=float, default=0.0001, help='use TTUR lr rate for Adam')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='use TTUR lr rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing testing.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'fashion'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True, help='use tensorboard to record the loss')

    # Path
    parser.add_argument('--dataroot', type=str, default='../data', help='dataset path')
    parser.add_argument('--log_path', type=str, default='./logs', help='the output log path')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint', help='model save path')
    parser.add_argument('--sample_path', type=str, default='./samples', help='the generated sample saved path')

    # Step size
    parser.add_argument('--log_step', type=int, default=500, help='every default{500} epoch save to the log')
    parser.add_argument('--sample_step', type=int, default=500, help='every default{500} epoch save the generated images and real images')
    parser.add_argument('--model_save_step', type=int, default=500, help='every default{500} epoch save the model state dict')


    return parser.parse_args()

# %%
def main(config):
    # data loader 
    data_loader = getdDataset(config)

    # delete the exists path
    del_folder(config.sample_path, config.version)
    del_folder(config.log_path, config.version)
    del_folder(config.model_save_path, config.version)

    # create directories if not exist
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.model_save_path, config.version)
    
    # save sample images
    make_folder(config.sample_path, config.version + '/real_images')
    make_folder(config.sample_path, config.version + '/fake_images')

    if config.train:
        if config.seed is not None:
            # in order to make the model repeatable, the first step is to set random seeds, 
            # and the second step is to the trainer function.
            random.seed(config.seed)
            torch.manual_seed(config.seed)

            # for the current configuration, so as to optimize the operation efficiency.
            cudnn.benchmark = True
            # ensure the every time the same input returns the same result.
            cudnn.deterministic = True

        if config.model == 'cgan':
            trainer = Trainer_cgan(data_loader, config)
        trainer.train()
    
# %% 
if __name__ == '__main__':
    config = get_parameters()
    pprint.pprint(config)
    main(config)