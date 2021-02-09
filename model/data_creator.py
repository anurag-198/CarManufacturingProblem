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
from collections import deque
import matplotlib.pyplot as plt
import operator
import copy
# Project specific inputs
from environment import Environment
from replayBuffer import ReplayBuffer
from dgnAgent import Agent
from trainer import Trainer
from plotter import Plotter
import functions as fc


# create a test_sequence in the length if inpur parameter number
# return the name of the new testsequence
# method will never overwrite existing testsequences but index them with _i instead
def create_test_sequences(number, length, KIND_CARS):
    name = 'N:' + str(number) +'_L:' + str(length) + '_KC:' + str(KIND_CARS)
    pathname = fc.get_path() + 'testsequences/'
    content = os.listdir(pathname)

    i = 1
    new_name = name
    while new_name + '.testsequence' in content:
        new_name = name + '_' + str(i)
        i += 1

    f = open(pathname + '/' + new_name + ".testsequence", "w")
    f.write(str(number) + ' ' + str(length) + "\n")

    for _ in range(number*length):
        f.write(str(random.randint(0, KIND_CARS-1)) + "\n")

    f.close()

    return new_name

def load_test_sequences(name):
    pathname = fc.get_path()

    f = open(pathname + 'testsequences/' + name + ".testsequence", "r")
    dim = f.readline().split()
    rows = int(dim[0])
    cols = int(dim[1])
    test_sequences = np.zeros((rows * cols), int)
    for i, line in enumerate(f):
        test_sequences[i] = int(line)
    f.close()
    test_sequences = test_sequences.reshape((rows, cols))
    test_sequences = torch.tensor(test_sequences).long()

    return test_sequences


# def create_buffer_resets(name, number, KIND_CARS, NUM_LINES, CAPACITY_LINES, OUTPUT_SEQUENCE_LENGTH):
#     pathname = fc.get_path()
#
#     res = []
#     ret = []
#     for _ in range(number):
#         buffer = np.random.randint(0, KIND_CARS, NUM_LINES * CAPACITY_LINES)
#         output_sequence = np.random.randint(0, KIND_CARS, OUTPUT_SEQUENCE_LENGTH)
#
#         ret.append((buffer, output_sequence))
#         res.append(np.concatenate((buffer, output_sequence)))
#
#     f = open(pathname + '/' + name + ".test", "w")
#     f.write(str(number) + " " + str(NUM_LINES) + " " + str(CAPACITY_LINES) + " " + str(OUTPUT_SEQUENCE_LENGTH) +  "\n")
#
#     for seq in res:
#         for num in seq:
#             f.write(str(num) + "\n")
#     f.close()
#
#     return None
#
# def load_buffer_resets(name):
#     pathname = fc.get_path()
#
#
#     f = open(pathname + '/' + name + ".test", "r")
#     dim = f.readline().split()
#
#     number = int(dim[0])
#     NUM_LINES = int(dim[1])
#     CAPACITY_LINES = int(dim[2])
#     OUTPUT_SEQUENCE_LENGTH = int(dim[3])
#
#     interm = np.zeros((number* (NUM_LINES * CAPACITY_LINES + OUTPUT_SEQUENCE_LENGTH)))
#
#
#     for i, line in enumerate(f):
#         interm[i] = int(line)
#
#     buffers = []
#     output_sequences = []
#
#     interm = interm.reshape((number, NUM_LINES * CAPACITY_LINES + OUTPUT_SEQUENCE_LENGTH))
#     for entry in interm:
#         buffer = entry[0:(NUM_LINES*CAPACITY_LINES)]
#         output_sequence = entry[NUM_LINES*CAPACITY_LINES:]
#
#         if len(output_sequence) != OUTPUT_SEQUENCE_LENGTH:
#             print("error, length should be equal")
#             return None
#
#         buffers.append(buffer.astype(int))
#         output_sequences.append(output_sequence.astype(int))
#
#
#     return buffers, output_sequences
#
