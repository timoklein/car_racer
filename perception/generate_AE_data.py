
import numpy as np
from numpy import ndarray
import multiprocessing as mp
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

_BATCH_SIZE = 16
_NUM_BATCHES = 1
_TIME_STEPS = 150
_RENDER = True


def generate_action(prev_action: ndarray) -> ndarray:
    """
    Generates random actions in the gym environment CarRacer.
    The actions are biased towards acceleration to induce exploration of the environment.  
    
    ## Parameters:  
    
    - **prev_action** *(ndarray)*: Array with 3 elements representing the previous action.     
    
    ## Output:  
    
    - **action** *(ndarray)*: Array with 3 elements representing the new sampled action.
    """
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action*mask


def simulate_batch(batch_num: int):
    """
    Generates batches of observations from the Carracer environment.  
    
    ## Parameters:  
    
    - **batch_num** *(int)*:  Number of the current batch being generated.  
    """
    env = CarRacing()

    obs_data = []
    action_data = []
    action = env.action_space.sample()
    for i_episode in range(_BATCH_SIZE):
        observation = env.reset()
        # Make the Car start at random positions in the race-track
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])

        obs_sequence = []

        for _ in range(_TIME_STEPS):
            if _RENDER:
                env.render()

            action = generate_action(action)

            observation, reward, done, info = env.step(action)

            obs_data.append(observation)

    print("Saving dataset for batch {}".format(batch_num))
    # Change this to an appropriate directory for you
    np.save('/home/timo/DataSets/carracer_images/obs_data_AE_{}'.format(batch_num), obs_data)
    
    env.close()


if __name__ == "__main__":
    print("Generating data for env CarRacing-v0")

    with mp.Pool(mp.cpu_count()) as p:
        p.map(simulate_batch, range(_NUM_BATCHES))