import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
import logging
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

class SAC(object):
    """
    SAC (Soft actor critic) algorithm.
    Objects contain the Q networks and optimizers, as well as functions to 
    safe and load models, selecting actions and perform updates for the objective functions.

    
    ## Parameters:  
    
    - **num_inputs** *(int)*: dimension of input (In this case number of variables of the latent representation)
    - **action_space**: action space of environment (E.g. for car racer: Box(3,) which means that the action space has 3 actions that are continuous.)
    - **args**: namespace with needed arguments (such as discount factor, used policy and temperature parameter)
 
    """
    def __init__(self,
                 action_space,
                 policy: str = "Gaussian",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 lr: float = 0.0003,
                 alpha: float = 0.2,
                 automatic_temperature_tuning: bool = False,
                 batch_size: int = 256,
                 hidden_size: int = 256,
                 target_update_interval: int = 1,
                 input_dim: int = 32):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_temperature_tuning = automatic_temperature_tuning

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bs = batch_size

        self.critic = QNetwork(input_dim, action_space.shape[0], hidden_size).to(DEVICE)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(input_dim, action_space.shape[0], hidden_size).to(DEVICE)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_temperature_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(DEVICE)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)


            self.policy = GaussianPolicy(input_dim, action_space.shape[0], hidden_size).to(DEVICE)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_temperature_tuning = False
            self.policy = DeterministicPolicy(input_dim, action_space.shape[0], hidden_size).to(DEVICE)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        settings = (f"INITIALIZING SAC ALGORITHM WITH {self.policy_type} POLICY"
                    f"\nRunning on: {DEVICE}"
                    f"\nSettings: Automatic Temperature tuning = {self.automatic_temperature_tuning}, Update Interval = {self.target_update_interval}"
                    f"\nParameters: Learning rate = {self.lr}, Batch Size = {self.bs} Gamma = {self.gamma}, Tau = {self.tau}, Alpha = {self.alpha}"
                    f"\nArchitecture: Input dimension = {self.input_dim}, Hidden layer dimension = {self.hidden_size}"
                    "\n--------------------------")
        
        print(settings)

    def select_action(self, state, eval=False):
        #TODO Marius input "eval" nochmal genau nachvollziehen, was das ist
        #TODO Marius mit Timo absprechen wie genau shape von arrays etc dokumentiert werden
        """
        Returns an action based on a given state from policy. 
        
        ## Input:  
        
        - **state** *(type)*: State of the environment. In our case latent representation with 32 variables.  
        - **eval** *(boolean)*: indicates whether to evaluate or not  
        
        ##Output:  
        
        - **action[0]**: [1,3] Selected action based on policy. Array with [s: steering,a: acceleration, d:deceleartion] coefficients
        """

        state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        return action[0]



    def update_parameters(self, memory, batch_size, updates):
        """
        Computes loss and updates parameters of objective functions (Q functions, policy and alpha).
        
        ## Input:  
        
        - **memory**: instance of class ReplayMemory  
        - **batch_size** *(int)*: batch size that shall be sampled from memory
        - **updates**: indicates the number of the update steps already done 
        
        ## Output:  
        
        - **qf1_loss.item()**: loss of first q function 
        - **qf2_loss.item()**: loss of second q function
        - **policy_loss.item()**: loss of policy
        - **alpha_loss.item()**: loss of alpha
        - **alpha_tlogs.item()**: alpha tlogs (For TensorboardX logs)
        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(DEVICE)
        next_state_batch = torch.FloatTensor(next_state_batch).to(DEVICE)
        action_batch = torch.FloatTensor(action_batch).to(DEVICE)
        reward_batch = torch.FloatTensor(reward_batch).to(DEVICE).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(DEVICE).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_temperature_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name: str, identifier: str, suffix: str = ".pt", actor_path=None, critic_path=None):
        path = Path("models/")
        if not path.exists():
            path.mkdir()

        if actor_path is None:
            actor_path = (path/f"sac_actor_{env_name}_{identifier}").with_suffix(suffix)
        if critic_path is None:
            critic_path = (path/f"sac_critic_{env_name}_{identifier}").with_suffix(suffix)
        print(f"Saving models to {actor_path} and {critic_path}")
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    """
    Insert your description here.  
    
    ## Parameters:  
    
    - **param1** *(type)*:  
    
    ## Input:  
    
    - **Input 1** *(shapes)*:  
    
    ## Output:  
    
    - **Output 1** *(shapes)*:  
    """
    def load_model(self, actor_path, critic_path):
        print(f"Loading models from {actor_path} and {critic_path}")
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

