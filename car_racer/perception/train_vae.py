import torch
import torchvision
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm
import numpy as np
from typing import Tuple
import logging

from autoencoders import ConvBetaVAE
from utils import PathOrStr

# Some global constants and Paramters
logging.basicConfig(level=logging.INFO, style='$')
BETA=3
"""Global constant beta for the variational autoencoder."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


def get_data(path_to_x: PathOrStr, path_to_y: PathOrStr) -> Tuple[DataLoader, DataLoader]:
    """
    Function for getting data into form ready for model consumption. Loads two tensordatasets x and y.
    Then shuffles the indices and returns a training and a validation DataLoader object.  
    
    **Parameters**:  
    
    - *path_to_x* (PathOrStr): Path to the data.  
    - *path_to_y* (PathOrStr): Path to the training labels.  

    **Input**:  
    
    - *Tensors* of the shape: [N: # samples, 3: in_channels, H: in_height, w: in_width].   
    
    **Output**:  
    
    - *Tuple*: Tuple consisting of two DataLoaders with the shape (train_loader, validation_loader).
    """
    x = torch.load(path_to_x)
    y = torch.load(path_to_y)

    data = TensorDataset(x,y)

    # sanity check
    assert len(x) == len(y), "X and Y don't have the same numer of elements!"

    # get training indices and split up
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2*num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # generate sampler for training and validation indices
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # instantiate dataloaders
    train_loader = DataLoader(data, batch_size=32, sampler=train_sampler)
    valid_loader = DataLoader(data, batch_size=32, sampler=valid_sampler)

    return train_loader, valid_loader



def loss_fn(x_hat: Tensor, y: Tensor, mu: Tensor, logvar: Tensor) -> float:
    """
    Loss function of β-VAE, check https://arxiv.org/pdf/1804.03599.pdf
    or https://dylandjian.github.io/world-models/.   
    
    **Parameters**:  
    
    - *x_hat* (Tensor): Reconstruced image by the model.  
    - *y* (Tensor): True target grayscale image.  
    - *mu* (Tensor): Mean vector of the autoencoder model.
    - *logvar* (Tensor): Log(Variance) of the autoencoder model.  

    **Input**:  
    
    - *Tensor* x_hat of shape: [N: batch size, 3: in_channels, H: in_height, w: in_width].   
    - *Tensor* y of shape: [N: batch size, 3: in_channels, H: in_height, w: in_width].  
    - *Tensor* mu of shape: Mean vector of the autoencoder model.
    - *Tensor* logvar of shape: Log(Variance) of the autoencoder model. 
    
    **Output**:  
    
    - *loss*: Scalarized loss.
    """

    batch_size = y.size()[0]
    loss = F.mse_loss(x_hat, y, reduction="sum")

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss /= batch_size
    kld /= batch_size
    return loss + BETA * kld.sum()


def train_batch(vae, optimizer, x: Tensor, y: Tensor) -> float:
    """
    Trains the VAE for a single batch of training images.  
    
    **Parameters**:  
    
    - *vae* (ConvBetaVAE): Autoencoder model to be trained.  
    - *optimizer*: PyTorch optimizer.
    - *x* (Tensor): Training example.  
    - *y* (Tensor): Training labels.  
    
    **Output**:  
    
    - *Loss* (float): Scalar loss value.  
    """

    optimizer.zero_grad()

    x_hat, mu, logvar = vae(x)
    loss = loss_fn(x_hat, y, mu, logvar)

    loss.backward()
    optimizer.step()

    return float(loss)


def train_model(epochs: int = 20) -> None:
    """
    Trains the variational autoencoder for specified number of epochs using Adam as optimizer.
    The function implements a simple variant of learning rate scheduling where the learning rate
    is divided by 10 after certain loss thresholds are reached.  
    
    **Parameters**:  
    
    - *epochs* (int=20): Number of training epochs.    
    """
    LR = 1e-03
    scheduler = optim.Adam(vae.parameters(), lr=LR)
    logging.info(f"Optimizing with learning rate: {LR}")
    total_ite = 0
    flag_first_cycle = True
    flag_second_cycle = True
    for e in tqdm(range(epochs)):
        
        batch_loss_train, running_loss_train = [], []
        batch_loss_valid, running_loss_valid = [], []
        
        print(f"Training for epoch: {e}")
        for i,data_train in enumerate(train_loader):
            x,y = data_train
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            loss = train_batch(vae, scheduler, x,y)
            running_loss_train.append(loss)
            
            ## Print running loss
            if i % 10 == 0:
                batch_loss_train.append(sum(running_loss_train)/len(running_loss_train))
                running_loss_train = []
            
            total_ite += 1

        if len(batch_loss_train) > 0:
            avg_train_loss = sum(batch_loss_train)/len(batch_loss_train)
            logging.info(f"[TRAIN] Iteration: {total_ite} | Average loss : {avg_train_loss} | LR: {LR}")
        
        print(f"Validation for epoch: {e}")
        # validation of the model
        vae.eval()
        for i,data_valid in enumerate(valid_loader):
            x,y = data_train
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            with torch.no_grad():
                # need to only calculate the loss here!
                x_hat, mu, logvar = vae(x)
                loss = loss_fn(x_hat, y, mu, logvar)
                running_loss_valid.append(loss)
                
                ## Print running loss
            if i % 10 == 0:
                batch_loss_valid.append(sum(running_loss_valid)/len(running_loss_valid))
                running_loss_valid = []

        if len(batch_loss_valid) > 0:
            avg_valid_loss = sum(batch_loss_valid)/len(batch_loss_valid)
            logging.info(f"[VALID] Iteration: {total_ite} | Average loss : {avg_valid_loss} | LR: {LR}")
                
        vae.train()

        if avg_valid_loss < 50 and avg_train_loss < 50 and flag_first_cycle:
            LR /= 10
            scheduler = optim.Adam(vae.parameters(), lr=LR)
            logging.info(f"Optimizing with learning rate: {LR}")
            flag_first_cycle = False
        elif avg_valid_loss < 45 and avg_train_loss < 50 and flag_second_cycle:
            LR /= 10
            scheduler = optim.Adam(vae.parameters(), lr=LR)
            logging.info(f"Optimizing with learning rate: {LR}")
            flag_second_cycle = False


if __name__ == '__main__':
    # get data
    path_to_x = "/home/jupyter/tutorials/praktikum_ml/color_data.pt"
    path_to_y = "/home/jupyter/tutorials/praktikum_ml/grayscale_data.pt"
    train_loader, valid_loader = get_data(path_to_x, path_to_y)


    # instantiate model and optimizer
    vae = ConvBetaVAE()
    vae.to(DEVICE)
    # As of now you can't use this in PyTorch together with ADAM
    # https://github.com/pytorch/pytorch/issues/19003
    # # optimizer = optim.RMSprop(vae.parameters())
    # # scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-04, max_lr=1e-02, 
    #                                         step_size_up=2000, cycle_momentum=False,
    #                                         mode="triangular2")
    
    train_model(20)

    # save model
    vae.cpu()
    torch.save(vae.state_dict(), 'weights.pt')
