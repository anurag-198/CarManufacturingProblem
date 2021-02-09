import sys
sys.path.append("../")
sys.path.append("../model")
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from IPython.display import clear_output
import network_loader



def get_path():
    path = os.path.dirname(sys.argv[0]) + '/'
    if not '/toy-model/' in path:
        path = os.getcwd() + '/'
    print(path)
    if path == '':
        path = './'
    #else:
        #index_start = path.index('/test/')
        #index_end = index_start + len('/toy-model/')
        #path = path[:index_start]
    return "/media/shadowwalker/DATA/study/RIL1/code/carmanufacturing"

def one_hot(states, kind_cars):
    length = len(states)
    occ = np.ones(length)
    occ[states == -1] = 0
    res = np.zeros((length, kind_cars))
    for i, state in enumerate(states):
        if occ[i]:
            res[i, state] = 1

    return occ, res



def linearize(states, kind_cars):
    length = len(states)
    occ, states = one_hot(states, kind_cars)
    states = states.reshape(length * (kind_cars))
    return np.concatenate([occ, states])

class Agent_wrapper():
    
    def __init__(self, net, KIND_CARS):
        self.net = net
        self.kind_cars = KIND_CARS
        
    def act(self, env, eval_mode = False, n_determined = None):
        if not n_determined:
            state = env.get_state()
        else:
            state = env.get_randomized_state(n_determined = n_determined)
        state = linearize(state, self.kind_cars)
        state = torch.tensor(state).float()

        with torch.no_grad():
            action_values = self.net(state).numpy()

        if eval_mode:
            possible_actions = env.possible_actions()
            for i in range(len(action_values)):
                if i not in possible_actions:
                    action_values[i] = - float("inf")

        return np.argmax(action_values)

    def q_values(self, state):
        state = linearize(state, self.kind_cars)
        state = torch.tensor(state).float()

        with torch.no_grad():
            return self.net(state).numpy()

class RandomPlayer():

    def act(self, env, eval_mode = False):
        actions = env.possible_actions()
        return np.random.choice(actions)


def add_agents(num_lines, capacity_lines, KIND_CARS, MA = False, SA = False, CP = False, Curr = False):

    choosen_options = int(MA) + int(SA) + int(CP) + int(Curr)
    if choosen_options == 0 or choosen_options > 1:
        print("you must choose exactly one option")
        return None

    front_agents = []
    back_agents = []
    for _ in range(10):
        front_agents.append([])
        back_agents.append([])

    num_lines_string = 'NL:' + str(num_lines)
    capacity_lines_string = 'CL:' + str(capacity_lines)

    pathname = get_path() + 'results/'

    for subdir in os.listdir(pathname):
        if subdir[0] == '.':
            continue
        sign = subdir[0:2]
        if MA and sign != 'MA':
            continue
        if SA and sign != 'SA':
            continue
        if CP and sign != 'CP':
            continue
        if Curr and sign != 'CC':
            continue
        for filename in os.listdir(pathname + subdir):
            if filename[-4:] == '.pth':
                if num_lines_string in filename and capacity_lines_string in filename:
                    network_name = pathname + subdir + '/' + filename
                    net = network_loader.load_network(network_name)
                    agent = Agent_wrapper(net, KIND_CARS)

                    input_size_index = filename.index('I:') + 2
                    input_size = int(filename[input_size_index])

                    if MA or Curr:
                        if 'Front' in filename:
                            front_agents[input_size].append((filename[:-4], agent))
                        else:
                            if 'Back' in filename:
                                back_agents[input_size].append((filename[:-4], agent))
                            else:
                                print('Something went wrong with ', filename)
                    else:
                        front_agents[input_size].append((filename[:-4], agent))
    if MA or Curr:
        return front_agents, back_agents
    else:
        return front_agents
