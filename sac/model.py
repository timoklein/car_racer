import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()

        # input
        self.conv1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)),
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)),
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)),
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)),
        ]))
        self.linear1_1 = nn.Linear(4096 + num_actions, 1028)
        self.linear1_2 = nn.Linear(1028, 1024)
        self.linear1_3 = nn.Linear(1024, 1)

        # Q2 architecture
        self.conv2 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            ("relu1", nn.LeakyReLU(0.2)),
            ("block1", ConvBlock(32, 64, 4, stride=2, padding=1, slope=0.2)),
            ("block2", ConvBlock(64, 128, 4, stride=2, padding=1, slope=0.2)),
            ("block3", ConvBlock(128, 256, 4, stride=2, padding=1, slope=0.2)),
        ]))
        self.linear2_1 = nn.Linear(4096 + num_actions, 1028)
        self.linear2_2 = nn.Linear(1028, 1024)
        self.linear2_3 = nn.Linear(1024, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state1 = self.conv1(state)
        state2 = self.conv2(state)
        state1 = state1.flatten()
        state2 = state2.flatten()
        xu1 = torch.cat([state1, action], 1)
        xu2 = torch.cat([state2, action], 1)
        
        x1 = F.relu(self.linear1_1(xu1))
        x1 = F.relu(self.linear1_2(x1))
        x1 = self.linear1_3(x1)

        x2 = F.relu(self.linear2_1(xu2))
        x2 = F.relu(self.linear2_2(x2))
        x2 = self.linear2_3(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean


    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean
    
