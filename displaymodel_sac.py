import itertools
import datetime

import numpy as np
import gym
import torch
from torch.utils.tensorboard import SummaryWriter


from sac.sac import SAC
from sac.replay_memory import ReplayMemory
from perception.utils import load_model, process_observation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""



def main(seed: int = 69,
          batch_size: int = 512,
          episodes: int = 100,
          path_to_actor: str = "models/sac_actor_carracer_klein_6_24_18.pt",
          path_to_critic: str = "models/sac_critic_carracer_klein_6_24_18.pt"):
    # Environment
    env = gym.make("CarRacing-v0")
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Agent
    agent = SAC(env.action_space,
                policy = "Gaussian",
                gamma = 0.99,
                tau = 0.005,
                lr = 0.0003,
                alpha = 0.2,
                automatic_temperature_tuning = False,
                batch_size = batch_size,
                hidden_size = 256,
                target_update_interval = 1,
                input_dim = 32)

    #load models and throws error if the paths are wrong
    agent.load_model(path_to_actor, path_to_critic)

    avg_reward = 0.
    rewards = []

    for i_episode in range(episodes):
        episode_reward = 0
        # Get initial observation 
        state = env.reset()
        state = process_observation(state)
        state = encoder.sample(state)
        done = False
        for t in range(1000):
            # render the environment at each step
            env.render()
            # move the car using the policy actions
            action = agent.select_action(state, eval=True)
            state, reward, done, _ = env.step(action)
            state = process_observation(state)
            state = encoder.sample(state)
            episode_reward += reward


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        rewards.append(episode_reward)
        avg_reward += episode_reward
    # Close the rendering

    np.save("rewards.npy", rewards)
    env.close()
    avg_reward /= episodes
    print("-"*40)
    print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}")
    print("-"*40)   


 

if __name__ == "__main__":
    encoder = load_model("models/VAE_weights.pt", vae=True)
    encoder.to(DEVICE)
    main()
