import random
import numpy as np
from environment import IN, OUT, Environment
from wrapper import SingleAgentWrapper
import time

MAX_SCORE = 900

class Rollout():

    def __init__(self):
        self.is_random = False

    def act(self, env):
        return 0

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0
        eps_counter = 10
        eps = 1.0
        lookahead = node.lookahead
        sim_length = 0

        while (not done) and lookahead <= max_lookahead:
            lookahead += int(next_env.get_player() == OUT)
            action = self.act(next_env)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += np.power(eps, eps_counter)*reward
            eps_counter += 1
            sim_length += 1

        if sim_length == 0:
            return ret
        else:
            return ret #/ sim_length

class RandomRollout(Rollout):

    def __init__(self):
        super().__init__()
        self.is_random = True

    def act(self, env):
        possible_actions = env.possible_actions()
        return random.choice(possible_actions)

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0

        while not done:
            action = self.act(next_env)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward

        return ret

class CheatyMARollout(Rollout):

    def __init__(self, back_wrapper, front_wrapper):
        super().__init__()
        self.front = front_wrapper
        self.back = back_wrapper

    def act(self, env):
        #assert isinstance(env, Environment)
        if env.get_player() == OUT:
            return self.back.act(env, eval_mode = True)
        else:
            return self.front.act(env, eval_mode = True)

    def max_q_value(self, state, player):
        if player == OUT:
            return np.max(self.back.q_values(state))
        else:
            return np.max(self.front.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0
        lookahead = node.lookahead

        next_state = None
        while (not done) and lookahead <= max_lookahead:
            #assert next_env.get_player() == OUT
            lookahead += int(next_env.get_player() == OUT)
            action = self.act(next_env)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward


        if done:
            return ret
        else:
            return ret + self.max_q_value(next_state, player = next_env.get_player())

class MARollout(Rollout):

    def __init__(self, back_wrapper, front_wrapper):
        super().__init__()
        self.front = front_wrapper
        self.back = back_wrapper

    def act(self, env, n_determined):
        #assert isinstance(env, Environment)
        if env.get_player() == OUT:
            return self.back.act(env, eval_mode = True, n_determined = n_determined)
        else:
            return self.front.act(env, eval_mode = True, n_determined = n_determined)

    def max_q_value(self, state, player):
        if player == OUT:
            return np.max(self.back.q_values(state))
        else:
            return np.max(self.front.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0
        lookahead = node.lookahead

        next_state = None
        while (not done) and lookahead <= max_lookahead:
            #assert next_env.get_player() == OUT
            lookahead += int(next_env.get_player() == OUT)
            n_determined = max(max_lookahead - lookahead, 0)
            action = self.act(next_env, n_determined)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward


        if done:
            return ret
        else:
            next_state = next_env.get_randomized_state(n_determined = 0)
            return ret + self.max_q_value(next_state, player = next_env.get_player())

class CheatySARollout(Rollout):

    def __init__(self, sa_wrapper):
        super().__init__()
        self.sa_wrapper = sa_wrapper

    def act(self, env):
        #assert isinstance(env, SingleAgentWrapper)
        return self.sa_wrapper.act(env, eval_mode = True)

    def max_q_value(self, state):
        return np.max(self.sa_wrapper.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0
        lookahead = node.lookahead

        #n_determined = max(0, max_lookahead - lookahead)
        #return sum([self.max_q_value(next_env.get_randomized_state(n_determined = n_determined)) for _ in range(10)]) / 10

        next_state = None
        while (not done) and lookahead <= max_lookahead:
            #assert next_env.get_player() == OUT
            lookahead += int(next_env.get_player() == OUT)
            action = self.act(next_env)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward

        if done:
            return ret
        else:
            return ret + self.max_q_value(next_state)

class SARollout(Rollout):

    def __init__(self, sa_wrapper):
        super().__init__()
        self.sa_wrapper = sa_wrapper

    def act(self, env, n_determined):
        #assert isinstance(env, SingleAgentWrapper)
        return self.sa_wrapper.act(env, eval_mode = True, n_determined = n_determined)

    def max_q_value(self, state):
        return np.max(self.sa_wrapper.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env
        done = node.env.done
        ret = 0
        lookahead = node.lookahead

        #n_determined = max(0, max_lookahead - lookahead)
        #return sum([self.max_q_value(next_env.get_randomized_state(n_determined = n_determined)) for _ in range(10)]) / 10

        next_state = None
        while (not done) and lookahead <= max_lookahead:
            #assert next_env.get_player() == OUT
            lookahead += int(next_env.get_player() == OUT)
            n_determined = max(max_lookahead - lookahead, 0)
            action = self.act(next_env, n_determined)
            next_env, (reward, next_state, done) = next_env.light_step(action)
            ret += reward

        next_state = next_env.get_randomized_state(n_determined = 0)

        if done:
            return ret
        else:
            return ret + self.max_q_value(next_state)


class PseudoSARollout(Rollout):

    def __init__(self, sa_wrapper, sample_size = 8):
        super().__init__()
        self.sa_wrapper = sa_wrapper
        self.sample_size = sample_size

    def max_q_value(self, state):
        return np.max(self.sa_wrapper.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env

        lookahead = node.lookahead
        n_determined = max(0, max_lookahead - lookahead)
        sample_size = self.sample_size * (node.input_window_length - n_determined + 1)
        #sample_size = self.sample_size
        return sum([self.max_q_value(next_env.get_randomized_state(n_determined = n_determined)) for _ in range(sample_size)]) / sample_size

class PseudoMARollout(Rollout):

    def __init__(self, back_wrapper, front_wrapper, sample_size = 8):
        super().__init__()
        self.sample_size = sample_size
        self.front = front_wrapper
        self.back = back_wrapper

    def max_q_value(self, state, player):
        if player == OUT:
            return np.max(self.back.q_values(state))
        else:
            return np.max(self.front.q_values(state))

    def simulate(self, node, max_lookahead):

        next_env = node.env

        lookahead = node.lookahead
        n_determined = max(0, max_lookahead - lookahead)
        sample_size = self.sample_size * (node.input_window_length - n_determined + 1)
        #sample_size = self.sample_size
        return sum([self.max_q_value(next_env.get_randomized_state(n_determined = n_determined), next_env.get_player()) for _ in range(sample_size)]) / sample_size


class Node():

    def __init__(self, env, father, reward = 0, number = 'r', act_id = None, rollout = None):



        self.env = env
        self.father = father
        self.root = father == None
        self.explored = False
        self.children = []
        self.number = number

        self.reward = reward
        self.N = 0
        self.V = 0
        self.U = 0
        self.act_id = act_id

        self.input_window_length = env.input_window_length

    
        if self.root:
            self.rollout = rollout if rollout else RandomRollout()
        else:
            self.rollout = father.rollout

        # mcts should not have more information than input_window_size for a fair comparison
        # increase this for successors generated by "OUT" actions
        # compare to max_lookahead to make fair cutoff
        # i.e. if node.lookahed > environment.input_window_length then stop growing the tree
        self.lookahead = 0 if self.root else father.lookahead + int(isinstance(env,SingleAgentWrapper) or father.env.get_player() == OUT)  # This should be used again if in the fill_ratio < 1 scenario


    def backprop(self, Nc, Vc):
        self.V = (self.N * self.V + Nc * Vc ) / (self.N + Nc)
        self.N += Nc

        if not self.root:
            self.father.backprop(Nc,Vc)


    def simulate(self, max_lookahead):
        return self.rollout.simulate(self, max_lookahead)

    def explore(self):
        # assert len(self.env.possible_actions()) > 0, 'DEAD END : Buffer : %s Player : %s PA : %s' % (str(self.env.buffer), str(self.env.player), str(self.env.possible_actions()))
        for i, action in enumerate(self.env.possible_actions()):
            oenv, (reward, next_state, done) = self.env.light_step(action)
            self.children.append(Node(oenv, self, reward, self.number + str(i), action))
        self.explored = True

    def update(self):
        if self.root:
            return self.U
        C = 0.5
        self.U = (self.V + self.reward) + C*np.sqrt(np.log(self.father.N)/(self.N + 1))
        return self.U

    def visit(self, max_lookahead):
        if (self.rollout.is_random and self.N == 0) or (self.lookahead <= max_lookahead and self.N == 0):
            new_value = self.simulate(max_lookahead)
            self.V = new_value
            self.N += 1
            # print(self.number, "first visit")

            if not self.root:
                self.father.backprop(1, new_value+self.reward)

            return 1

        else:

            if self.env.done or self.lookahead > max_lookahead:
                self.father.backprop(1, self.reward)
                # print(self.number, "final state")
                return 0

            if not self.explored:
                # print(self.number, "explore")
                self.explore()

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

            return successor.visit(max_lookahead)

    def show(self, d = 0):

        indent = '-' * d

        def prnt(s):
            print(indent+s)

        prnt('|----')
        prnt('|DEPTH ' + str(d) + '.')
        prnt("|N: " + str(self.N))
        prnt("|V: " + str(self.V))
        prnt("|R: " + str(self.reward))
        self.update()
        prnt("|U: " + str(self.U))
        prnt("|EXPL: " + str(self.explored))
        prnt("|C: " + str(self.children))
        prnt("|ROOT: " + str(self.root))
        prnt("|DONE:" + str(self.env.done))


    def print_tree(self, d = 0):
        #child_numbers = []
        self.show(d)
        for child in self.children:
            child.print_tree(d + 1)
            #child_numbers.append((child.number, child.N))
        #pr = str(self.number) + " - " + str(self.N) + " , children: "
        #for cn in child_numbers:
        #    pr = pr + str(cn[0]) + ", "
        #print(pr)

    def final_value(self):
        return self.N

class SPNode(Node):

    def __init__(self, env, father, reward = 0, number = 'r', act_id = None, rollout = None):
        super().__init__(env, father, reward, number, act_id, rollout = rollout)
        self.ssV = 0

    def backprop(self, Nc, Vc):
        self.N += Nc
        self.V = self.V + (Vc - self.V) / self.N
        self.ssV = self.ssV + np.power(Vc,2)
        if not self.root:
            self.father.backprop(Nc,Vc)

    def simulate(self, max_lookahead):
        return self.rollout.simulate(self, max_lookahead)

    def explore(self):
        for i, action in enumerate(self.env.possible_actions()):
            oenv, (reward, next_state, done) = self.env.light_step(action)
            self.children.append(SPNode(oenv, self, reward, self.number + str(i), action))
        self.explored = True

    def update(self):
        if self.root:
            return self.U
        C = 0.5
        # TODO WHICH VALUES MAKE SENSE here, paper is unclear about this
        D = 0
        ucb_term = C * np.sqrt(np.log(self.father.N) / (self.N + 1))
        variance_estimate = np.sqrt((self.ssV - self.N * np.power(self.V,2) + D) / (self.N + 1))
        self.U = (self.V + self.reward) + ucb_term + variance_estimate
        return self.U

    def visit(self, max_lookahead):
        if self.lookahead <= max_lookahead and self.N == 0:
            new_value = self.simulate(max_lookahead)
            self.V = new_value
            self.ssV += new_value**2
            self.N += 1
            # print(self.number, "first visit")

            if not self.root:
                self.father.backprop(1, new_value + self.reward)

            return 1

        else:

            if not self.root and (self.env.done or self.lookahead > max_lookahead):
                self.father.backprop(1, self.reward)
                # print(self.number, "final state")
                return 0

            if not self.explored:
                # print(self.number, "explore")
                self.explore()

            elif self.env.done:
                return 0

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
            #successor = random.choice(possible_successors)
            successor = random.choice(possible_successors)
            # print(self.number, " number: " , self.N, " visit->", successor.number)

            return successor.visit(max_lookahead)

    def final_value(self):
        return self.N

class MCTS():

    def __init__(self, env, depth = 100, root = None, InitNode = SPNode, limit_lookahead = True, rollout = None, time_limit = 0.05):

        self.depth = depth
        self.max_lookahead = env.input_window_length + env.counter if limit_lookahead else float("inf")
        self.InitNode = InitNode
        self.limit_lookahead = limit_lookahead
        self.rollout = rollout
        self.time_limit = time_limit

        if root == None:
            self.root = InitNode(env, None, rollout = rollout)
        else:
            self.root = root

        self.termination_horizon = self.root.N + self.depth

    def reset(self, new_env, root = None):
        self.__init__(new_env, depth = self.depth, root = root, InitNode = self.InitNode, limit_lookahead = self.limit_lookahead, rollout = self.rollout, time_limit = self.time_limit)

    def act(self, env, inherit = True, eval_mode = True):

        if env.counter == 0 and env.get_player() == OUT:
            self.reset(env)
        else:
            self.reset(env, self.root)


        start = time.time()
        self.root.visit(self.max_lookahead)
        self.root.visit(self.max_lookahead)
        while self.root.N < self.termination_horizon and time.time() - start < self.time_limit:
            self.root.visit(self.max_lookahead)

        actions = []
        for child in self.root.children:
            actions.append(child.final_value())

        #if len(self.root.children) == 0:
        #    print('DEAD END? : RootBuffer : %s\n Player : %s\n PA : %s\n' % (str(self.root.env.env.buffer), str(self.root.env.get_player()), str(self.root.env.possible_actions())))
        #    self.root.show()
        #    assert False

        res = np.argmax(actions)
        child = self.root.children[res]
        act_id = child.act_id
        #print('Selected child with val ', self.root.children[res].final_value())
        if inherit:
            self.root = child
            self.father = None
            self.root.root = True
        return act_id

    def show_tree(self):
        self.root.print_tree()


