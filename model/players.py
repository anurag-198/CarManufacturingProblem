import sys
sys.path.append("../")
sys.path.append("../model")
import numpy as np
from model import environment
from trainer import Trainer
import data_creator as dc
import os
import functions as fc
import time

# Constants
KIND_CARS = 8
INPUT_SEQUENCE_LENGTH = 100
INPUT_WINDOW = 1
OUTPUT_SEQUENCE_LENGTH = 4
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

initial_ratio = 0.6

class RandomPlayer():

    def act(self, env, eval_mode = False):
        actions = env.possible_actions()
        return np.random.choice(actions)

def remove(list, sublist):

    for index, element in enumerate(list):
        if (element == sublist).all():
            list.pop(index)
            break

def maximums(list):
    max_distance = - 1
    result = []
    for (distance, a) in list:
        if distance < max_distance:
            pass
        else:
            if distance == max_distance:
                result.append(a)
            else:
                max_distance = distance
                result = [a]
    return result

class Lastseen_heuristics_out():

    def act(self, env, eval_mode = False):
        po = env.possible_actions()


        possible_results = []
        for a in po:
            car = env.get_buffer()[a][-1]

            if car in env.get_output():
                index = env.get_output().index(car)
                possible_results.append((index,a))

            else:
                possible_results.append((float("inf"), a))

        result = maximums(possible_results)

        return np.random.choice(result)

class Similarity_in_heuristics():

    def act(self, env, eval_mode = False):
        po = env.possible_actions()
        car = env.input_sequence[0].item()

        sims = np.zeros(env.num_lines)

        for i,line in enumerate(env.buffer):
            for stored in line:
                if car == stored:
                    sims[i] += 1

        for i in range(len(sims)):
            if i not in po:
                sims[i] = -float("inf")

        pa = []
        max_sim = 0
        for i, sim in enumerate(sims):
            if sim < max_sim:
                pass
            else:
                if sim == max_sim:
                    pa.append(i)
                else:
                    pa = [i]
                    max_sim = sim

        return np.random.choice(pa)

class Split_in_half_heuristics():

    def act(self, env, eval_mode = False):
        po = env.possible_actions()
        car = env.input_sequence[0].item()

        if len(po) == 1:
            return po[0]

        if env.num_lines == 2:
            if car < 4:
                return 0
            else:
                return 1

        else:
            # num_lines == 4
            if car < 2 and 0 in po:
                return 0
            if car < 4 and 1 in po:
                return 1
            if car < 6 and 2 in po:
                return 2
            if car < 8 and 3 in po:
                return 3

        return np.random.choice(po)
