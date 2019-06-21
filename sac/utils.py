import math
import torch

def soft_update(target, source, tau):
    """
    A soft parameter update is performed when update_parameters() in the SAC class is called. The parameters from the 
    Q network are copied to the target network according to the following formula:  
    *target network parameters = target network parameters * (1 - τ) + Q network parameters * τ*  
    
    ## Parameters:  
    
    - **target** *(QNetwork)*:  Target network (time-delayed version of the Q network)
    - **source** *(QNetwork)*:  Q network
    - **tau** *(float)*: Target smoothing coefficient
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    A hard parameter update is performed when the SAC class is instantiated. The parameters from the Q network are copied 
    to the target network.
    
    ## Parameters:  
    
    - **target** *(QNetwork)*:  Target network (time-delayed version of the Q network)
    - **source** *(QNetwork)*:  Q network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
