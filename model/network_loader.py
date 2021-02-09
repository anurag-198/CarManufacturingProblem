from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy



class N4(nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.fc1 = nn.Linear(layer_numbers[0], layer_numbers[1])
        self.fc2 = nn.Linear(layer_numbers[1], layer_numbers[2])
        self.fc3 = nn.Linear(layer_numbers[2], layer_numbers[3])
        self.fc4 = nn.Linear(layer_numbers[3], layer_numbers[4])
        self.fc5 = nn.Linear(layer_numbers[4], layer_numbers[5])


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

class N3(nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.fc1 = nn.Linear(layer_numbers[0], layer_numbers[1])
        self.fc2 = nn.Linear(layer_numbers[1], layer_numbers[2])
        self.fc3 = nn.Linear(layer_numbers[2], layer_numbers[3])
        self.fc4 = nn.Linear(layer_numbers[3], layer_numbers[4])


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x



class N2(nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.fc1 = nn.Linear(layer_numbers[0], layer_numbers[1])
        self.fc2 = nn.Linear(layer_numbers[1], layer_numbers[2])
        self.fc3 = nn.Linear(layer_numbers[2], layer_numbers[3])


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class N1(nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.fc1 = nn.Linear(layer_numbers[0], layer_numbers[1])
        self.fc2 = nn.Linear(layer_numbers[1], layer_numbers[2])


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def load_network(name):
    networks_classes = [N1,N2, N3, N4]
    possible_weights = [32,64,128,256]
    input_sizes = range(10)
    OUTPUT_SEQUENCE_LENGTH = 4
    num_lines = [2,4,6]
    CAPACITY_LINES = 3
    KIND_CARS = 8


    network_loaded = False
    for i,C in enumerate(networks_classes):
        for nl in num_lines:
            for input_size in input_sizes:
                input_weight = [(input_size + OUTPUT_SEQUENCE_LENGTH + nl * CAPACITY_LINES) * (KIND_CARS + 1)]
                for output_weight in [[nl], [nl*nl]]:
                    for w in possible_weights:
                        try:
                            intermediate_weights = (np.ones(i+1, int) * w).tolist()
                            weights = input_weight + intermediate_weights + output_weight
                            net = C(weights)
                            net.load_state_dict(torch.load(name))
                            net.eval()
                            return net
                        except:
                            pass
    print(name, "not loaded!")


