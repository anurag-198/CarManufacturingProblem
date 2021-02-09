import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import torch
from torch import optim
import torch.nn.functional as F
import sys
import random
import numpy as np
import operator
import copy
# Project specific inputs
from environment import Environment
from replayBuffer import ReplayBuffer
from dgnAgent import Agent
from trainer import Trainer
from plotter import Plotter
import data_creator
from mcts_expert import Node, MCTS
import functions as fc


KIND_CARS = 8
INPUT_SEQUENCE_LENGTH = 100
INPUT_WINDOW = 3
OUTPUT_SEQUENCE_LENGTH = 5
NUM_LINES = 2
CAPACITY_LINES = 3


# Constants Agent
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 1          # discount factor
TAU = 0.001#1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
SEED = 0


# define neural network
class Network(nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.fc1 = nn.Linear(layer_numbers[0], layer_numbers[1])
        self.fc2 = nn.Linear(layer_numbers[1], layer_numbers[2])
        self.fc3 = nn.Linear(layer_numbers[2], layer_numbers[3])
        # self.fc4 = nn.Linear(layer_numbers[3], layer_numbers[4])
        # self.fc5 = nn.Linear(layer_numbers[4], layer_numbers[5])


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        return x


env = Environment(INPUT_SEQUENCE_LENGTH, KIND_CARS, NUM_LINES, CAPACITY_LINES, OUTPUT_SEQUENCE_LENGTH, INPUT_WINDOW)
#
# pathname = fc.get_path()
# policy = pathname + '/results/I:3_O:5_N:126-64-64-2_NL:2_CL:3_W:100_KC:8/I:3_O:5_N:126-64-64-2_NL:2_CL:3_W:100_KC:8_0.pth'
# net.load_state_dict(torch.load(policy))
# net.eval()
# agent = Agent(net, BUFFER_SIZE, BATCH_SIZE, UPDATE_EVERY, GAMMA, TAU, LR, SEED)
#
# sum = 0
# done = False
# for i in range(INPUT_SEQUENCE_LENGTH):
#     mcts = MCTS(env, agent, KIND_CARS, 100)
#     act = mcts.execute()
#     # act = random.choice([0,1])
#     # print(i, act)
#     # for x in mcts.root.children:
#         # print(x.N, x.V, x.U)
#     reward, next_state, done = env.step(act)
#     sum += reward
#     print(i, "/", INPUT_SEQUENCE_LENGTH)
#     if done:
#         break
#
# print("Summe: " + str(sum))
# print(env.done, env.counter, i)
# mcts.show_tree()