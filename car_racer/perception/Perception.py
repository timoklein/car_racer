import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from collections import OrderedDict

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 3x64x64
        self.convlayer1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)

        # input: 32x32x32
        self.convblock1 = nn.Sequential(OrderedDict([
            ("Conv", nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(64)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 64x16x16
        self.convblock2 = nn.Sequential(OrderedDict([
            ("Conv", nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(128)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 128x8x8
        self.convblock3 = nn.Sequential(OrderedDict([
            ("Conv", nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(256)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 256x4x4
        self.convlayer2 = nn.Conv2d(256, 32, 4, stride=1, padding=0)
        # output: 32x1x1
        
        # input: 32x1x1
        self.deconvblock1 = nn.Sequential(OrderedDict([
            ("Deconv", nn.ConvTranspose2d(32, 256, 4, stride=1, padding=0)),
            ("BN", nn.BatchNorm2d(256)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 256x4x4
        self.deconvblock2 = nn.Sequential(OrderedDict([
            ("Deconv", nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(128)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 128x8x8
        self.deconvblock3 = nn.Sequential(OrderedDict([
            ("Deconv", nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(64)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        # input: 64x16x16
        self.deconvblock4 = nn.Sequential(OrderedDict([
            ("Deconv", nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)),
            ("BN", nn.BatchNorm2d(32)),
            ("ReLu", nn.LeakyReLU(0.2))
        ]))

        #input: 32x32x32
        self.deconvlayer1 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        
        #output: 1x64x64 image (grayscale)

        #----------------------------------------------------------------------------
        def encode(self, x):
            """
            Ecnodes a 3x64x64 input image into a 32x1x1 latent representation.
            """
            x = self.relu1(self.convlayer1(x))
            x = self.convblock1(x)
            x = self.convblock2(x)
            x = self.convblock3(x)
            return self.convlayer2(x)

        def decode(self, x):
            """
            Decodes a 32x1x1 vector into a 1x64x64 grayscale image.
            """
            x = self.deconvblock1(x)
            x = self.deconvblock2(x)
            x = self.deconvblock3(x)
            x = self.deconvblock4(x)
            return F.sigmoid(self.deconvlayer1(x))

        def forward(self, x):
            """
            Forward pass is used to train the model.
            """
            x = self.encode(x)
            return self.decode(x)   

