import argparse
import gym
import numpy as np
import itertools
import torch
import logging
from sac.sac import SAC
from sac.replay_memory import ReplayMemory
from perception.utils import load_model, process_observation
from torch.utils.tensorboard import SummaryWriter
import datetime
from perception.generate_AE_data import generate_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=69, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                    help='maximum number of steps (default: 2000001)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: True)')
parser.add_argument('--encoder_output_dimension', type=int, default=32,
                    help='Dimension of the encoder output (default: 32)')
parser.add_argument('--save_models', type=bool, default=True,
                    help='Option to save models to ./models (default: True)')
parser.add_argument('--load_models', type=bool, default=True,
                    help='Option to load models (default: True) -> Use path_to_actor and path_to_critic to specify a path')
parser.add_argument('--path_to_actor', default="./models/sac_actor_carracer_latest",
                    help='Path to previously saved actor model (default: "./models/sac_actor_carracer_latest")')
parser.add_argument('--path_to_critic', default="./models/sac_critic_carracer_latest", 
                    help='Path to previously saved critic model (default: "./models/sac_critic_carracer_latest")')
args = parser.parse_args()

def main():
    #TODO Marius Docu hier noch anpassen
    """
    Training loop. Consists of: 
                                -Setting up environment, agent and memory
                                -loading models (if possible);
                                -training
                                -evaluation (every x episodes)
                                -saving models

    """
    # Environment
    env = gym.make("CarRacing-v0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Agent
    agent = SAC(args.encoder_output_dimension, env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0

    #Tensorboard
    writer = SummaryWriter(log_dir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "NochSinnvolleBennenungUeberlegen",#args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))



                                                        

    if args.load_models:
        try:
            agent.load_model(args.path_to_actor, args.path_to_critic)
        except FileNotFoundError:
            print("Could not find models. Starting training without models:")

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        state = process_observation(state)
        state = encoder.sample(state)

        #First action random
        action = env.action_space.sample()

        while not done:
            if args.start_steps > total_numsteps:
                #Instead of using totally random action, we use random action biased towards acceleration
                #action = env.action_space.sample()  # Sample random action
                action = generate_action(action)
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for _ in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,args.batch_size,updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            next_state = process_observation(next_state)
            next_state = encoder.sample(next_state)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        if i_episode % 50 == 0 and args.eval == True:
            avg_reward = 0.
            episodes = 10

            if args.save_models: agent.save_model('carracer', 'latest')

            for _ in range(episodes):
                state = env.reset()
                state = process_observation(state)
                state = encoder.sample(state)

                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, eval=True)

                    next_state, reward, done, _ = env.step(action)
                    next_state = process_observation(next_state)
                    next_state = encoder.sample(next_state)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            if args.save_models: agent.save_model('carracer', 'latest')

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

            memory.save("buffer")

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, style='$')
    encoder = load_model("/fzi/ids/michel/no_backup/WeigthsAutoencoder/VAE_weights.pt", vae=True)
    encoder.to(DEVICE)
    main()