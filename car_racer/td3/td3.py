import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    """
    Feedforward Actor network for the TD3 algorithm. Consists of three linear layers with relu activation.
    The last layer uses tanh activation.  

    **Parameters**:  

    - *state_dim* (int): Dimensionality of the environment's state space.  
       The state space should be represented as a vector of floats.
    - *action_dim* (int): Dimensionality of the environment's action space.  
       The action space should be a vector of float values.
    - *max_action* (float): Highest possible action space value.    

    **Input**:  

    - Input 1: [shapes]  

    **Output**:  

    - Output 1: [shapes]  
    """
    def __init__(self, state_dim: int, action_dim: int, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    
    def forward(self, x):
        """
        Forward pass. The last layer uses a tanh activation and is multiplied by max_action.
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x


class Critic(nn.Module):
    """
    Insert your description here.  

    **Parameters**:  
    
    - *param1* (type):  
    
    **Input**:  
    
    - Input 1: [shapes]  
    
    **Output**:  
    
    - Output 1: [shapes]  
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1 


class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, 
              replay_buffer, 
              iterations, 
              batch_size: int = 100, 
              discount: float = 0.99, 
              tau: float = 0.005, 
              policy_noise: float = 0.2, 
              noise_clip: float = 0.5, 
              policy_freq: int = 2):

        for it in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(DEVICE)
            action = torch.FloatTensor(u).to(DEVICE)
            next_state = torch.FloatTensor(y).to(DEVICE)
            done = torch.FloatTensor(1 - d).to(DEVICE)
            reward = torch.FloatTensor(r).to(DEVICE)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(DEVICE)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                # tau is a parameter for computing a weighted average of
                # parameters and target parameters when doing a target update
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))

