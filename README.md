# CGAN-PyTorch
Pytorch implementation of condtions GANs

## Overview
This repository contains an Pytorch implementation of Conditional GAN.
With full coments and my code style.

## About DCGAN
If you're new to CGAN, here's an abstract straight from the paper[1]:

## Dataset 
- MNIST
`python3 main.py --dataset mnist --channels 1`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3`

## Implement
``` python

```

## Usage
- MNSIT
`python3 main.py --dataset mnist --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- FashionMNIST
`python3 main.py --dataset fashion --channels 1 --version [version] --batch_size [] >logs/[log_path]`
- Cifar10
`python3 main.py --dataset cifar10 --channels 3 -version [version] --batch_size [] >logs/[log_path]`

## Tensorboard 
The G loss and D loss record by the tensorboard in the folder, /logs.
``` python
tensorboard --logdir logs/
```

## FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

For the FID, I use the pytorch implement of this repository. [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid)

For the 10k epochs training on different dataset, compare with about 10000 samples, I get the FID: 

| dataset | DCGAN |
| ---- | ---- |
| MNIST | null |
| FASHION-MNIST | null | 
| CIFAR10 | null |

> :warning: I dont konw if the FID is right or not, because I cant get the lowwer score like the paper or the other people get it. 
## Network structure
``` python

```
## Result
- MNIST  
<!-- ![9900_MNSIT](img/9900_MNIST.png) -->
- CIFAR10  
<!-- ![9900_cifar10](img/9900_cifar10.png) -->
- Fashion-MNIST
<!-- ![9900_fashion](img/9900_fashion.png) -->
## Reference
1. [CGAN](https://arxiv.org/abs/1411.1784)
