# -*- coding: utf-8 -*-
"""
Challenges and Opportunities of Data Driven Advance Classification for Hard
Rock TBM excavations

---- script to paper
DOI: XXXXXXX

Discriminator and Generator implementation from DCGAN paper:
Radford et.al., 2015 - Unsupervised Representation Learning with Deep
Convolutional Generative Adversarial Networks

Removed Sigmoid() from Discriminator output (as proposed in the WGAN Paper):
Arjovsky et.al., 2017 - Wassertein GAN

@author: Paul Unterlass
Created on Thu Jun 13 2024
"""

import torch
import torch.nn as nn

# =============================================================================
# Dicriminator Neural Network
# =============================================================================

class Discriminator(nn.Module):
    def __init__(self, channel, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: batch size (N) x channels (4) x vector size (4096)
            nn.Conv1d(
                channel, features_d, kernel_size=8, stride=4, padding=2 # Input: 1x4096 Output: 1+(4096-8+2*2)/4 = 1024
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 8, 4, 2), # Input: 1x1024, Output: 1+(1024-8+2*2)/4 = 256
            self._block(features_d * 2, features_d * 4, 8, 4, 2), # Input: 1x256, Output: 1+(256-8+2*2)/4 = 64
            self._block(features_d * 4, features_d * 8, 8, 4, 2), # Input: 1x64, Output: 1+(64-8+2*2)/4 = 16
            self._block(features_d * 8, features_d * 16, 8, 4, 2), # Input: 1x16, Output: 1+(16-8+2*2)/4 = 4
                        
            nn.Conv1d(features_d * 16, 1, kernel_size=4, stride=1, padding=0), # Input: 1x4, Output: 1+(4-4+2*0)/1 = 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
    
# =============================================================================
# Generator Neural Network
# =============================================================================

class Generator(nn.Module):
    def __init__(self, channels_noise, channel, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: batch_size (N) x channels_noise x 1
            self._block(channels_noise, features_g * 32, 4, 1, 0),  # Input: 1x1, Output: (4-1)*1+4-2*0+1 = 8 --> 1x8
            self._block(features_g * 32, features_g * 16, 8, 4, 2),  # Input: 1x8, Output: (8-1)*4+8-2*2 = 32
            self._block(features_g * 16, features_g * 8, 8, 4, 2),  # Input: 1x32, Ouput: (32-1)*4+8-2*2 = 128
            self._block(features_g * 8, features_g * 4, 8, 4, 2),  # Input: 1x128, Output: (128-1)*4+8-2*2 = 512
            self._block(features_g * 4, features_g * 2, 8, 4, 2),  # Input: 1x256, Output: (256-1)*4+8-2*2 = 1024
            
            nn.ConvTranspose1d(
                features_g*2, channel, kernel_size=8, stride=4, padding=2
            ), # Input: 1x1024, Output: 4096
            # Output: batch_size (N) x channels x features_g
            nn.Sigmoid(), # [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    # Radford et.al., 2015 - Unsupervised Representation Learning with Deep
    # Convolutional Generative Adversarial Networks
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# =============================================================================
# test Neural Networks
# =============================================================================

def test():
    N, in_channels, H = 32, 4, 4096
    noise_dim = 100
    x = torch.randn((N, in_channels, H)) # [64, 1, 64]
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1)) # [64, 100, 1]
    assert gen(z).shape == (N, in_channels, H), "Generator test failed"
    print('NN test succeeded')
    return disc, gen, x, z

disc, gen, x, z = test()
