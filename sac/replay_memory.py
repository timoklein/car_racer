import random
import numpy as np
from pathlib import Path
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def save(self, path):
        print("Saving replay buffer...")
        save_dir = Path("memory/")
        if not save_dir.exists():
            save_dir.mkdir()
        with open((save_dir/path).with_suffix(".pkl"), "wb") as fp:
            pickle.dump(self.buffer, fp)

    def __len__(self):
        return len(self.buffer)
