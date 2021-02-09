# Replay Buffer used for Experience Replay
from collections import deque
import random
import torch


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        if done:
            done_value = 1
        else:
            done_value = 0
        self.memory.append([state, action, reward, next_state, done_value])

    def sample(self):
        samples = (random.sample(self.memory, self.batch_size))
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for sample in samples:
            state, action, reward, next_state, done = sample

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.tensor(states).float()
        actions = torch.LongTensor(actions)
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float()

        return [states, actions, rewards, next_states, dones]

    def __len__(self):
        return len(self.memory)