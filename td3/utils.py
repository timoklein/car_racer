import torch
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

class ReplayBuffer:
    """
    Simple replay Buffer for the TD3 algorithm. The code is based on  
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py.  
    Taken from https://github.com/sfujim/TD3.   
    
    ## Parameters:
    
    - **max_size** *(int=1e6)*: Replay Buffer size  
    """
    def __init__(self, max_size: int = 1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """
        ## Input:  
    
        - **Tuple** *(state, next_state, action, reward, done)*:  Experience sample.
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """
        ## Output:  
    
        - **Tuple** *(state batch, next_state batch, action batch, reward batch, done batch)*:  
        Batch of experience samples. Each element is a numpy array.
        """
        ind = torch.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(torch.Tensor(X, copy=False))
            y.append(torch.Tensor(Y, copy=False))
            u.append(torch.Tensor(U, copy=False))
            r.append(torch.Tensor(R, copy=False))
            d.append(torch.Tensor(D, copy=False))

        state = torch.FloatTensor(x).to(DEVICE)
        action = torch.FloatTensor(u).to(DEVICE)
        next_state = torch.FloatTensor(y).to(DEVICE)
        done = torch.FloatTensor(1 - d).to(DEVICE)
        reward = torch.FloatTensor(r).to(DEVICE)

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

