import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from collections import OrderedDict



class ConvBlock(nn.Module):
    """
    Convolutional building block for neural models with sensible standard values.  
    Consists of a 2D convolutional layer, a 2D batchnorm layer and leaky relu activation.  
    Further information: https://pytorch.org/docs/stable/nn.html#conv2d.  

    ## Parameters:  

    - **in_channels** *(int)*:  number of input channels for the 2D convolution.  
    - **out_channels** *(int)*: number of output channels for 2D convolution.  
    - **kernel_size** *(int)*:  square kernel height and width for 2D convonlution.  
    - **stride** *(int=2)*:     stride of the 2D convolution.  
    - **padding** *(int=1)*:    padding of the 2D convolution.  
    - **slope** *(float=0.2)*:  negative slope for the leaky relu activation.  
    """
    def __init__(self, in_channels: int, 
                        out_channels: int,
                        kernel_size: int, 
                        stride: int = 2, 
                        padding: int = 1, 
                        slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=slope)
    
    def forward(self, x):
        """
        ## Inputs:  

        - **Tensor** *(N: batch size, C: in_channels, H: in_height, W: in_width)*:  
            Batch of input images. 

        ## Outputs:  

        - **Tensor** *(N: batch size, C: out_channels, H: out_height, W: out_width)*:  
            Convolved batch of input images. 
        """
        return self.relu(self.bn(self.conv(x)))


class DeConvBlock(nn.Module):
    """
    Deconvolutional building block for various convolutional models with sensible standard values.  
    Consists of a 2D transposed convolution layer, a 2D batchnorm layer and leaky relu activation.  
    Further information: https://pytorch.org/docs/stable/nn.html#convtranspose2d.  

    ## Parameters:  

    - **in_channels** *(int)*:  number of input channels for the 2D deconvolution.  
    - **out_channels** *(int)*: number of output channels for 2D deconvolution.  
    - **kernel_size** *(int)*:  square kernel height and width for 2D deconvonlution.  
    - **stride** *(int=2)*:     stride of the 2D deconvolution.  
    - **padding** *(int=1)*:    padding of the 2D deconvolution.  
    - **slope** *(float=0.2)*:  negative slope for the leaky relu activation.  
    """
    def __init__(self, in_channels: int, 
                        out_channels: int,
                        kernel_size: int, 
                        stride: int = 2, 
                        padding: int = 1, 
                        slope: float = 0.2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=slope)
    
    def forward(self, x):
        """
        ## Inputs:  

        - **Tensor** *(N: batch size, C: in_channels, H: in_height, w: in_width)*:  
            Batch of input images.  

        ## Output:  

        - **Tensor** *(N: batch size, C: out_channels, H: out_height, w: out_width)*:  
            Batch of input images with transposed convolution applied.
        """
        return self.relu(self.bn(self.deconv(x)))


class ConvAE(nn.Module):
    """
    A simple convolutional autoencoder. Processes RGB images in tensor form and reconstructs
    3 channel grayscale image tensors from a 32 dimensional latent space.  
    """
    def __init__(self):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)),
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)),
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)),
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)),
            ("conv2", nn.Conv2d(256, 32, 4, stride=1, padding=0))
        ]))
        
        # decoder
        self.decoder = nn.Sequential(OrderedDict([
            ("deconv1", DeConvBlock(32, 256, 4, stride=1, padding=0, slope=0.2)),
            ("deconv2", DeConvBlock(256, 128, 4, stride=2, padding=1, slope=0.2)),
            ("deconv3", DeConvBlock(128, 64, 4, stride=2, padding=1, slope=0.2)),
            ("deconv4", DeConvBlock(64, 32, 4, stride=2, padding=1, slope=0.2)),
            ("convt1", nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))
        ]))


    def sample(self, x):
        """
        Encodes a 3x64x64 input image into a latent representation with 32 variables.
        """
        x = torch.tanh(self.encoder(x))
        x_no_grad = x.detach().cpu()
        return x_no_grad.squeeze().numpy()
    

    def forward(self, x):
        """
        Forward pass for model training.  
        Consists of encoding and decoding operations applied sequentially.  

        ## Inputs:  

        - **Tensor** *(N: batch size, 3: # of in_channels, 64: height, 64: width)*:  
            Input image.
        
        ## Outputs:  

        - **Tensor** *(N: batch size, 3: # of out_channels, 64: height, 64: width)*:  
            Reconstructed Image.
        """
        x = torch.tanh(self.encoder(x))
        return torch.sigmoid(self.decoder(x))  



class ConvBetaVAE(nn.Module):
    """
    Convolutional beta-VAE implementation.
    The autoencoder is trained to reconstruct grayscale images from RGB inputs. Original paper: https://openreview.net/pdf?id=Sy2fzU9gl.  
    The code this implementation is based on can be found here:  
    https://dylandjian.github.io/world-models/.    
    
    ## Parameters:  
    
    - **z_dim** *(int=32)*: Number of latent variables.    
    """
    def __init__(self, z_dim: int = 32):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)), #32x32x32
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)), # 64x16x16
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)), # 128x8x8
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)), # 256x4x4
        ]))

        ## Latent representation of mean and std
        # 256x4x4 = 4096
        self.fc1 = nn.Linear(4096, z_dim)
        self.fc2 = nn.Linear(4096, z_dim)
        self.fc3 = nn.Linear(z_dim, 4096)

        # decoder
        self.decoder = nn.Sequential(OrderedDict([
            ("deconv1", DeConvBlock(4096, 256, 4, stride=1, padding=0, slope=0.2)),
            ("deconv2", DeConvBlock(256, 128, 4, stride=2, padding=1, slope=0.2)),
            ("deconv3", DeConvBlock(128, 64, 4, stride=2, padding=1, slope=0.2)),
            ("deconv4", DeConvBlock(64, 32, 4, stride=2, padding=1, slope=0.2)),
            ("convt1", nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))
        ]))


    def encode(self, x):
        """
        Encoding pass of the model.
        """
        x = self.encoder(x)
        x = x.view(-1, 4096)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        """
        Apply reparametrization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        """
        Decode latent representation.
        """
        z = self.fc3(z).view(-1, 256 * 4 * 4, 1, 1)
        z = self.decoder(z)
        return torch.sigmoid(z)
    
    def sample(self, x):
        """
        Compress input and sample latent representation.
        """
        # encode x
        x = self.encoder(x).view(-1, 4096)

        # get mu and logvar from input
        mu, logvar = self.fc1(x), self.fc2(x)

        # generate and return sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = z.detach().cpu()
        return z.squeeze().numpy()


    def forward(self, x, encode: bool = False, mean: bool = True):
        """
        Forward pass for the model.  

        ## Inputs:  
    
        - **Tensor** *(N: batch size, C: in_channels, H: in_height, w: in_width)*:  
            Represents the batch of input RGB images.

        ## Outputs:  

        - **Tensor** *(N: batch size, C: in_channels, H: in_height, w: in_width)*:  
            Represents the batch of output grayscale images.   
        - **Tensor** *(batch_size, z_dim)*: Batch of means for the latent variables.    
        - **Tensor** *(batch_size, z_dim)*: Batch of logvars for the latent variables.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar

