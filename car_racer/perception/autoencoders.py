import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from collections import OrderedDict

BETA=3

class ConvAE(nn.Module):
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


#-------------------------------------------------------------------------------------
# Simple VAE implementation
# TODO: Add Batchnorm Layers
# TODO: Change to LeakyRelu
# TODO: Understand and change latent representation code accordingly



class ConvBetaVAE(nn.Module):
    def __init__(self, input_shape, z_dim):
        super().__init__()

        ## Encoder
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        ## Latent representation of mean and std
        self.fc1 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc2 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc3 = nn.Linear(z_dim, 256 * 6 * 6)

        ## Decoder
        self.deconv1 = nn.ConvTranspose2d(256 * 6 *6, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 6, stride=2)

    #-------------------------------------------------------------------
    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 256 * 6 * 6)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        h = self.fc3(z).view(-1, 256 * 6 * 6, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        h = F.sigmoid(self.deconv5(h))
        return h


    def forward(self, x, encode=False, mean=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar


# loss function for training the beta-VAE
def loss_beta_VAE(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss /= batch_size
    kld /= batch_size
    return loss + BETA * kld.sum()

