import torch
import numpy as np
from collections import deque
import copy
import math
import random
from torch.distributions import Gamma

FAIL_REWARD = -100
POSITVE = 100

IN = 1
OUT = -1


class Environment():
    def __init__(self, game_length, kind_cars, num_lines, capacity_lines, output_sequence_length, input_window_length, episode, initial_ratio = 0.5, sequence = []):

        # save given variables
        self.length = game_length
        self.kind_cars = kind_cars
        self.num_lines = num_lines
        self.capacity_lines = capacity_lines
        self.output_sequence_length = output_sequence_length
        self.input_window_length = input_window_length
        self.initial_ratio = initial_ratio
        self.initial_fill = math.ceil((self.num_lines * self.capacity_lines) * initial_ratio)
        self.episode = episode

        # create variables
        self.counter = 0
        self.done = False

        # set player to in, as we sort in the cars. This is just for setup. Game will start with taking out cars.
        self.player = IN

        # initialise output sequence and buffer
        # not filled yet!
        self.output_sequence = deque(maxlen=output_sequence_length)
        interm = np.ones((num_lines, capacity_lines), int) * (-1)
        self.buffer = torch.tensor(interm).long()
        
        #No sequence given, fill everythin randomly
        if sequence == []:
            if (episode < 20000) :
                # Fill output randomly
                for _ in range(output_sequence_length):
                    self.output_sequence.append(torch.randint(0, kind_cars, (1,)).item())
                # Fill buffer randomly
                self.fill_buffer(random = True)
                # create random input sequence
                self.input_sequence = torch.randint(0, kind_cars, (self.length+self.input_window_length+1,))
            else :
                if (episode == 20001) :
                    print("episode 25001")
                self.input_sequence = []
                for i in range(output_sequence_length) :
                    if (i % 2) :
                        self.output_sequence.append(torch.randint(int(kind_cars/2), kind_cars, (1,)).item())
                    else :
                        self.output_sequence.append(torch.randint(0, int(kind_cars/2), (1,)).item()) 
                self.fill_buffer_newdis(random = True)
                
                for i in range(self.length+self.input_window_length+1) :
                    if (i % 2) :
                        self.input_sequence.append(torch.randint(int(kind_cars/2), kind_cars, (1,)).item())
                    else :
                        self.input_sequence.append(torch.randint(0, int(kind_cars/2), (1,)).item()) 
                    
                self.input_sequence = torch.tensor(self.input_sequence)
        else:
            # fill buffer with first numbers of sequences
            # remove them from sequence afterwards
            for i in range(output_sequence_length):
                self.output_sequence.append(sequence[i])
            sequence = sequence[output_sequence_length:]
            # fill buffer with sequence
            # remove used cars afterwards
            self.fill_buffer(random = False, sequence = sequence)
            sequence = sequence[self.initial_fill:]
            # store the remaining number of cars as input sequence
            self.input_sequence = torch.tensor(sequence)



        # now that the setup is done, the game will start with taking out a car from the buffer
        self.player = OUT

        # hard coded reward structure
        self.distances = np.ones(self.kind_cars)
        self.penalties = np.ones(self.kind_cars)

        self.distances[0:(int(self.kind_cars / 2))] = 2
        self.distances[(int(self.kind_cars / 2)):] = 3
        self.penalties[0:(int(self.kind_cars / 2))] = -5
        self.penalties[(int(self.kind_cars / 2)):] = -3


    def fill_buffer_newdis(self, random = False, sequence = []) :
        if random :
            for i in range(self.initial_fill):
                if (i % 2) :
                    car = torch.randint(int(self.kind_cars/2), self.kind_cars, (1,)).item()
                else :
                    car = torch.randint(0, int(self.kind_cars/2), (1,)).item()
                        
                line = np.random.choice(self.possible_actions())
                self.add_to_buffer(car, line)
                
    def fill_buffer(self, random = False, sequence = []):
        if not random and sequence == []:
            print("You most choose one option to fill the buffer!")
            return None
        if random:
            # randomly insert the needed number of random cars
            for i in range(self.initial_fill):
                car = np.random.randint(0, self.kind_cars)
                line = np.random.choice(self.possible_actions())
                self.add_to_buffer(car, line)
        else:
            # set seed! Results must be reproducable
            np.random.seed(np.sum(sequence))
            # insert the needed number of cars from the sequence
            for i in range(self.initial_fill):
                line = np.random.choice(self.possible_actions())
                car = sequence[i]
                self.add_to_buffer(car, line)


    def get_player(self):
        return self.player

    def switch_player(self):
        self.player *= (-1)

    def possible_actions(self):
        if self.done:
            return []
        res = []
        if self.get_player() == IN:
            for i in range(self.num_lines):
                line = self.buffer[i]
                if line[0] == -1:
                    res.append(i)
        else: # out mode
            for i in range(self.num_lines):
                line = self.buffer[i]
                if line[-1] != -1:
                    res.append(i)

        return res

    def reward(self):
        index = len(self.output_sequence) - 1
        car = self.output_sequence[index]

        low = int(np.max([0, index - self.distances[car]]))
        high = int(index + 1)
        window = torch.tensor(self.output_sequence)[low: high]

        failures = (window == car).sum().item() - 1
        reward = failures * self.penalties[car]

        return reward


    # light_step version for one agent mode.
    # check if this is still working for two agent before using it
    def light_step(self,action):
        if self.done:
            print("Already done, step won't have an effect")
            return None, None, None

        other_env = self.clone()
        res =  other_env.step(action)
        return other_env, res

    def add_to_buffer(self, car, action):
        line = self.buffer[action]
        # if line is not free at all, give penalty (and don't insert the car)
        if line[0] != -1:
            return False

            # manually check whether line is completely empty
        if line[-1] == -1:
            line[-1] = car
        else:
            # iterate over line and find next free spot
            free_spot = 0
            while True:
                if line[free_spot + 1] != -1:
                    break
                free_spot += 1
            line[free_spot] = car
        return True

    def step(self, action):
        #print(self.get_player())
        # first check if game is already over
        if self.done:
            print("Already done, step won't have an effect")
            return None, None, True
        # step switches between in and out
        if self.get_player() == IN:
            self.switch_player()
            self.counter += 1
            self.done = self.counter == self.length
            car = self.input_sequence[0].item()
            if not self.add_to_buffer(car, action):
                return (FAIL_REWARD, self.get_state(), self.done)

            self.input_sequence = self.input_sequence[1:]
            next_state = self.get_state()
            return 0, next_state, self.done

        else: # out mode
            self.switch_player()
            line = self.buffer[action]
            car = line[-1].item()
            # check whether the spot is empy -> invalid move, give penalty
            if car == -1:
                return (FAIL_REWARD, self.get_state(), self.done)
            # remove car from buffer and add an empty spot
            line = line[:-1]
            self.buffer[action] = torch.cat((torch.tensor([-1]), line))

            self.output_sequence.append(car)

            reward = self.reward()

            next_state = self.get_state()
            

            return reward, next_state, self.done



    def reset(self, episode, sequence = []):
        self.__init__(self.length, self.kind_cars, self.num_lines, self.capacity_lines, self.output_sequence_length, self.input_window_length, episode, self.initial_ratio, sequence)


    def get_state(self, readable = False):
        inp = self.input_sequence[0:(self.input_window_length)].tolist()
        while len(inp) < self.input_window_length:
            inp.insert(0, -1)
        if readable:
            return inp, self.buffer.tolist(), list(self.output_sequence)
        return inp + self.buffer.resize(self.capacity_lines * self.num_lines).tolist() + list(self.output_sequence)

    def get_randomized_state(self, readable = False, n_determined = 0):
        inp = self.input_sequence[0:(self.input_window_length)].tolist()
        inp = inp[0:n_determined] + [random.randrange(self.kind_cars) for _ in range(self.input_window_length - n_determined)]
        while len(inp) < self.input_window_length:
            inp.insert(0, -1)
        if readable:
            return inp, self.buffer.tolist(), list(self.output_sequence)
        return inp + self.buffer.resize(self.capacity_lines * self.num_lines).tolist() + list(self.output_sequence)

    def get_buffer(self):
        return self.buffer

    def get_output(self):
        return self.output_sequence

    
    def show(self):
        print("input: ", self.input_sequence)
        print("buffer: ", self.buffer)
        print("output: ", self.output_sequence)

    def get_stats(self):
        return self.length, self.kind_cars, self.num_lines, self.capacity_lines, self.output_sequence_length, self.input_window_length

    def clone(self):
        oe = Environment(self.length, self.kind_cars, self.num_lines, self.capacity_lines, self.output_sequence_length, self.input_window_length, self.initial_ratio)
        oe.buffer = copy.deepcopy(self.buffer)
        oe.output_sequence = copy.deepcopy(self.output_sequence)
        oe.input_sequence = copy.deepcopy(self.input_sequence)
        oe.counter = self.counter
        oe.done = self.done
        oe.player = self.player

        return oe

