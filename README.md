# Latent space reinforcement learning for autonomous driving (CarRacing-v0)  

<p align="center">
  <img src="assets/carracing.gif">
</p> 

This repository contains code to set up a reinforcement learning agent using the Soft Actor-Critic algorithm (SAC) [https://arxiv.org/pdf/1801.01290.pdf]. As perception module we choose a $`\beta`$-VAE [https://openreview.net/pdf?id=Sy2fzU9gl].  
Please note that while we provide a working implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) this algorithm has only been used for experimental reasons and is thus neither documented nor supported.  


## Getting started  
Clone the repo and install the dependencies from environment.yml (or requirements.txt). We recommend using miniconda to run our code as this project was developed using conda. train_sac.py is the main training script for our algorithm and display_sac.py shows the trained agent. 


## Project structure: 
    
    .
    ├── models    # Folder for autoencoder and SAC models
    |
    ├── perception   # Vision module
    |   ├── autoencoders.py    # Contains our encoder models
    |   ├── generate_AE_data.py    # Script to generate encoder training data
    |   └── utils.py    # helper Functions for encoder data handling
    |
    ├── sac    # Soft Actor-Critic implementation
    |   ├── model.py   # SAC neural net models
    |   ├── replay_memory.py   # Replay buffer
    |   ├── sac.py   # Main SAC implementation and training class
    |   └── utils.py   # Helper functions for SAC training
    |    
    ├── docs    # Documentation for the SAC+perception modules
    |   ├── perception    # Docs for the perception module
    │   |   └── xxx.html    # Html files
    |   |
    |   ├── sac    # Docs for the SAC algorithm
    |   |   └── xxx.html    # Html files
    |   |
    |   ├── displaymodel_sac.html    # SAC evaluation script
    |   ├── train_sac.html    # SAC training script
    |   └── train_vae.html    # VAE training script
    |
    ├── runs    # Example tensorboard training log 
    │   └── log folders    # Folders containing a training run's logs
    |       └── train logs    # tensorboard log files
    |
    ├── td3    # TD3 implementation (not supported)
    |   ├── td3.py    # main TD3 implementation (not supported)
    |   └── utils.py    # TD3 replay buffer (not supported)
    |
    ├── displaymodel_sac.py    # Evaluation script for a trained agent
    ├── train_sac_baselines.py    # Training script for the baselines agent
    ├── train_sac.py    # Training script for our SAC implementation
    ├── train_td3.py    # Training script for our SAC implementation
    ├── train_vae.py    # Training script for our VAE
    ├── environment.yml    # Conda environment
    ├── requirements.txt    # Pip requirements
    └── README.md     # This file


## Weights
The weights of our trained encoder models can be found in models and are named "weights.py" for the deterministic encoder and "VAE_weights.py" for the VAE.  
In addition to that we also provide some pretrained SAC agents. The best performing of those is "klein_6_24_18".  


## Results  
We provide a writeup containing the detailed results and further information about our setup here:
https://www.dropbox.com/s/ilkrxnwvfe493py/Latent_RL_paper.pdf?dl=0.


## Use this work  
Feel free to use our source code and results. 

	@article{LatReinforcementCARLA2019, 
		title={Latent Space Reinforcement Learning for Continous Control: An Evaluation in the Context of Autonomous Driving}, 
		author={Klein, Timo and Sigloch, Joan and Michel, Marius}, 
		year={2019}, 
	}