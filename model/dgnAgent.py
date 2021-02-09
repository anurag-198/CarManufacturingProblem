import torch
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import copy
from replayBuffer import ReplayBuffer

# Agent
class Agent():
    def __init__(self, q_network, buffer_size, batch_size, update_every, gamma, tau, lr,  seed):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.qnetwork_local = copy.deepcopy(q_network)
        self.qnetwork_target = copy.deepcopy(q_network)
        self.seed = random.seed(seed)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)

        self.t_step = 0

    def reset_memory(self):
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if (len(self.memory)) > self.batch_size:
                samples = self.memory.sample()
                self.learn(samples, self.gamma)

    def act(self, state, eps=0, eval_mode = True, pa = []):
        state = torch.tensor(state).float()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).numpy()

        if eval_mode:
            if random.random() > eps:
                for i in range(len(action_values)):
                    if i not in pa:
                        action_values[i] = - float("inf")
                return np.argmax(action_values)
            else:
                return random.choice(pa)
        else:
            if random.random() > eps:
                return np.argmax(action_values)
            else:
                return random.choice(range(len(action_values)))

    def learn(self, samples, gamma):
        states, actions, rewards, next_states, dones = samples

        q_values_next_states = self.qnetwork_target.forward(next_states).max(dim=1)[0]  # .unsqueeze(1)
        targets = rewards + (gamma * (q_values_next_states) * (1 - dones))
        q_values = self.qnetwork_local.forward(states)

        actions = actions.view(actions.size()[0], 1)
        predictions = torch.gather(q_values, 1, actions).view(actions.size()[0])

        loss = F.mse_loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # return buffer_size, batch_size, update_every, gamma, tau
    def get_stats(self):
        return self.buffer_size, self.batch_size, self.update_every, self.gamma, self.tau, self.lr
