import math

class Wrapper():

    def __init__(self,env):
        self.env = env

    def get_state(self, readable = False):
        return self.env.get_state(readable)

    def get_randomized_state(self, readable = False, n_determined = 0):
        return self.env.get_randomized_state(readable, n_determined)

    def get_player(self):
        return self.env.get_player()

    def possible_actions(self):
        return self.env.possible_actions()

    def reset(self, episode, sequence = []):
        return self.env.reset(episode, sequence)

    def get_stats(self):
        return self.env.get_stats()

    def show(self):
        return self.env.show()

    def get_buffer(self):
        return self.env.get_buffer()

    def get_output(self):
        return self.env.get_output()

class MultiAgentWrapper(Wrapper):

    def __init__(self, env, a1, a2):
        super().__init__(env)
        self.a1 = a1
        self.a1_auto = a1 != None
        self.a2 = a2
        self.a2_auto = a2 != None

        self.done = False
        self.states = []
        self.actions = []
        self.rewards = [0]



    def decide_next_turn(self, reward, former_action):

        self.states.append(self.env.get_state(readable=True))
        self.actions.append(former_action)

        if self.done:
            return reward, self.get_state(), self.done

        if self.env.get_player() == 1: # player one
            if self.a1_auto: # player one is auto player
                action = self.a1.act(self.env, eval_mode = True)
                next_reward, next_state, self.done = self.env.step(action)
                self.rewards.append(next_reward)
                return self.decide_next_turn(reward+next_reward, action)
            else: # player 1 , but manual player
                return reward, self.get_state(), self.done
        else: # player 2
            if self.a2_auto: # player 2 is auto player
                action = self.a2.act(self.env, eval_mode = True)
                next_reward, next_state, self.done = self.env.step(action)
                self.rewards.append(next_reward)
                return self.decide_next_turn(reward + next_reward, action)
            else: #player2, but manual player
                return reward, self.get_state(), self.done


    def step(self, action):
        reward, next_state, self.done = self.env.step(action)
        self.rewards.append(reward)
        return self.decide_next_turn(reward, action)


    def reset(self):
        self.done = False
        self.env.reset()
        self.states = []
        self.rewards = [0]

        return self.decide_next_turn(0, -1)

    def get_all(self):
        return self.states, self.actions, self.rewards


class CrossProductWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_lines = env.num_lines


    def decode(self, action):
        action_in = math.floor((action / self.num_lines))
        action_out = action % self.num_lines

        return action_out, action_in

    def encode(self, action_out, action_in):
        return action_in * self.num_lines + action_out


    def step(self, action):
        action_out, action_in = self.decode(action)

        reward, next_state, done = self.env.step(action_out)

        if done:
            return reward, next_state, done

        other_reward, next_state, done = self.env.step(action_in)

        return reward+other_reward, next_state, done

    def possible_actions(self):

        res = []
        possible_out = self.env.possible_actions()

        for a_out in possible_out:
            oe, (reward, state, done) = self.env.light_step(a_out)
            if reward == - 100:
                continue
            other_actions = oe.possible_actions()

            for a_in in other_actions:
                action = self.encode(action_out = a_out, action_in = a_in)
                res.append(action)

        return res

    def clone(self):
        oe = self.env.clone()
        ow = CrossProductWrapper(oe)

        return ow


class SingleAgentWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.done = env.done
        self.input_window_length = env.input_window_length
        self.counter = env.counter

    def reset(self, sequence = []):
        super().reset(sequence)
        self.input_window_length = self.env.input_window_length
        self.counter = self.env.counter
        self.done = self.env.done

    def step(self, action):
        reward, next_state, self.done = self.env.step(action)

        if self.done:
            return reward, next_state, self.done

        other_reward, next_state, self.done = self.env.step(action)
        self.counter = self.env.counter

        return reward + other_reward, next_state, self.done

    def clone(self):
        oe = self.env.clone()
        ow = SingleAgentWrapper(oe)

        return ow


    def light_step(self, action):
        ow = self.clone()
        res = ow.step(action)
        return ow, res

    def get_possible_actions(self):
        return env.get_possible_actions()
