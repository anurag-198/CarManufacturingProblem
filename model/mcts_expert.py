import random
import numpy as np
import functions as fc
from torch import nn
import torch

MAX_SCORE = 900

class Network(nn.Module):
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



class MCTS():
    def __init__(self, env, agent, kind_cars, depth = 100):
        self.depth = depth
        self.agent = agent
        self.kind_cars = kind_cars
        self.root = Node(env, None, self.agent, self.kind_cars)

    def execute(self):
        for i in range(self.depth):
            self.root.visit()

        actions = []
        for child in self.root.children:
            actions.append(child.N)

        return np.argmax(actions)

    def show_tree(self):
        self.root.print_tree()



class Node():

    def __init__(self, env, father, agent, kind_cars, reward = 0, number = 'r'):
        self.env = env
        self.father = father
        self.root = father == None
        self.explored = False
        self.children = []
        self.number = number
        self.agent = agent
        self.kind_cars = kind_cars

        self.reward = reward
        self.N = 0
        self.V = 0
        self.U = 0

    def backprop(self, Nc, Vc):
        self.V = (self.N * self.V + Nc * Vc ) / (self.N + Nc)
        self.N += Nc

        if not self.root:
            self.father.backprop(Nc,Vc)
    

    def simulate(self):
        next_env = self.env
        done = self.env.done
        ret = 0
        while (not done):
            state  = next_env.get_state()
            state = fc.linearize(state, self.kind_cars)
            action = self.agent.act(state, 0)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward

        return ret

    def explore(self):
        for i, action in enumerate(self.env.possible_actions()):
            oenv, (reward, next_state, done) = self.env.light_step(action)
            self.children.append(Node(oenv, self, self.agent, self.kind_cars, reward, self.number + str(i)))
        self.explored = True

    def update(self):
        if self.root:
            return self.U
        self.U = (self.V + self.reward) / MAX_SCORE + 1 + np.sqrt(self.father.N)/(self.N + 1)
        return self.U

    def visit(self):
        if self.N == 0:
            new_value = self.simulate()
            self.V = new_value
            self.N += 1
            # print(self.number, "first visit")

            if not self.root:
                self.father.backprop(1, new_value+self.reward)

        else:
            if not self.explored:
                # print(self.number, "explore")
                self.explore()

            if self.env.done:
                self.father.backprop(1, self.reward)
                # print(self.number, "final state")
                return

            possible_successors = []
            current_u = - float("inf")
            for i,child in enumerate(self.children):
                child_value = child.update()
                if child_value > current_u:
                    possible_successors = [child]
                    current_u = child_value
                else:
                    if child_value == current_u:
                        possible_successors.append(child)
            successor = random.choice(possible_successors)
            # print(self.number, " number: " , self.N, " visit->", successor.number)

            successor.visit()

    def show(self):
        print("N: " + str(self.N))
        print("V: " + str(self.V))
        print("reward: " + str(self.reward))
        self.update()
        print("U: " + str(self.U))
        print("explored: " + str(self.explored))
        print("children: " + str(self.children))
        print("root: " + str(self.root))
        print("done:" + str(self.env.done))


    def print_tree(self):
        child_numbers = []
        for child in self.children:
            child.print_tree()
            child_numbers.append((child.number, child.N))
        pr = str(self.number) + " - " + str(self.N) + " , children: "
        for cn in child_numbers:
            pr = pr + str(cn[0]) + ", "
        print(pr)
        self.show()
