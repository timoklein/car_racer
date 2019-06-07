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
    **Parameters**:  
    - *in_channels* (int):  number of input channels for the 2D convolution  
    - *out_channels* (int): number of output channels for 2D convolution  
    - *kernel_size* (int):  square kernel height and width for 2D convonlution  
    - *stride* (int=2):     stride of the 2D convolution  
    - *padding* (int=1):    padding of the 2D convolution  
    - *slope* (float=0.2):  negative slope for the leaky relu activation  
    **Input**:  
    - Tensor of shape: [N: batch size, C: in_channels, H: in_height, w: in_width].  
    **Output**:  
    - Tensor of shape: [N: batch size, C: out_channels, H: out_height, w: out_width]. 
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
        Forward pass.
        """
        return self.relu(self.bn(self.conv(x)))


class DeConvBlock(nn.Module):
    """
    Deconvolutional building block for various convolutional models with sensible standard values.  
    Consists of a 2D transposed convolution layer, a 2D batchnorm layer and leaky relu activation.  
    Further information: https://pytorch.org/docs/stable/nn.html#convtranspose2d.  
    **Parameters**:  
    - *in_channels* (int):  number of input channels for the 2D deconvolution  
    - *out_channels* (int): number of output channels for 2D deconvolution  
    - *kernel_size* (int):  square kernel height and width for 2D deconvonlution  
    - *stride* (int=2):     stride of the 2D deconvolution  
    - *padding* (int=1):    padding of the 2D deconvolution  
    - *slope* (float=0.2):  negative slope for the leaky relu activation  
    **Input**:  
    - Tensor of shape: [N: batch size, C: in_channels, H: in_height, w: in_width].  
    **Output**:  
    - Tensor of shape: [N: batch size, C: out_channels, H: out_height, w: out_width].  
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
        Forward pass.
        """
        return self.relu(self.bn(self.deconv(x)))


class ConvAE(nn.Module):
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

        #----------------------------------------------------------------------------
    def encode(self, x):
        """
        Encodes a 3x64x64 input image into a latent representation with 32 variables.
        """
        x = self.encoder(x)
        x_no_grad = x.detach()
        return x_no_grad.squeeze()

    def decode(self, x):
        """
        Decodes a latent vector with 32 elements into a 3x64x64 grayscale image.
        """
        return torch.sigmoid(self.decoder(x))

    def forward(self, x):
        """
        Forward pass for model training.  
        Consists of encoding and decoding operations applied sequentially.
        """
        x = self.encoder(x)
        return self.decoder(x)  

