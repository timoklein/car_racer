import numpy as np
import torch
import gym
import argparse
from pathlib import Path
import logging

from td3.utils import ReplayBuffer
from td3.td3 import TD3
from perception.utils import load_model, process_observation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

# TODO: Document this

def evaluate_policy(policy, eval_episodes=10):
    """
    Runs policy for X episodes and returns average reward.
    """
    avg_reward = 0.

    for _ in range(eval_episodes):
        obs = env.reset()
        obs = process_observation(obs)
        obs = encoder.encode(obs)
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            obs = process_observation(obs)
            obs = encoder.encode(obs)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("-"*40)
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
    print("-"*40)
    return avg_reward

# TODO: Document this
# TODO: Implement proper saving behaviour
# FIXME: Debug VAE pipeline
# TODO: Tune hyperparameters

def main(seed: int = 0,
         start_timesteps: int = 1e4,
         eval_freq: float = 5e3,
         max_timesteps: float = 1e6,
         save_models: bool = False,
         expl_noise: float = 0.1,
         batch_size: int = 100,
         discount: float = 0.99,
         tau: float = 0.005,
         policy_noise: float = 0.2,
         noise_clip: float = 0.5,
         policy_freq: int = 15):
         
    file_name = f"TD3_{seed}"
    logging.info("-"*40)
    logging.info("Settings: " + file_name)
    logging.info("-"*40)

    if not Path("./results").exists():
        Path("./results").mkdir()
    if save_models and not Path("./pytorch_models").exists():
        Path("./pytorch_models").mkdir()

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = 32 # latent space dimenions
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)] 

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True 

    while total_timesteps < max_timesteps:
        if done: 
            if total_timesteps != 0: 
                logging.info(f"Total T: {total_timesteps} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward}")
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # Evaluate episode
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(policy))

                if save_models: policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations) 

            # Reset environment
            obs = env.reset()
            obs = process_observation(obs)
            obs = encoder.encode(obs)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
		
        # Select action randomly or according to policy
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if expl_noise != 0: 
                action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        new_obs = process_observation(new_obs)
        new_obs = encoder.encode(new_obs)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
		
    # Final evaluation 
    evaluations.append(evaluate_policy(policy))
    if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, style='$')
    encoder = load_model("/home/timo/Documents/KIT/4SEM/0Praktikum_ML/weights.pt")
    env = gym.make("CarRacing-v0")
    main()

