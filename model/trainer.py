from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from IPython.display import clear_output
import functions as fc
from las import Look_ahead_search as Las
from wrapper import MultiAgentWrapper as MAW



class Trainer():

    def __init__(self, checkpoint_name, kind_cars, gamma):
        self.kind_cars = kind_cars
        self.gamma = gamma
        self.best_score = -float("inf")
        pathname = fc.get_path()
        counter = 0
        dirname = checkpoint_name + "_" + str(counter)
        while dirname in os.listdir(pathname + '/results'):
            counter += 1
            dirname = checkpoint_name + "_" + str(counter)
        self.checkpoint_name = dirname
        self.run_number = counter

    def save_settings(self, env, agent, multi_agent = False, curricular = None):
        pathname = fc.get_path()

        # get settings from env and agent
        input_sequence_length, kind_cars, num_lines, capacity_lines, output_sequence_length, input_window_length = env.get_stats()
        buffer_size, batch_size, update_every, gamma, tau, lr = agent.get_stats()

        # check if folder exists. create it otherwise

        f = open("/media/shadowwalker/DATA/study/RIL1/code/carmanufacturing/test/results/CP_NoVal/test.settings", "a")
        f.write("Gamma " + str(self.gamma) +"\n")
        f.write("Buffer Size " + str(buffer_size) +"\n")
        f.write("Batch Size " + str(batch_size) +"\n")
        f.write("Update Every " + str(update_every) +"\n")
        f.write("Tau " + str(tau) +"\n")
        f.write("LR " + str(lr) +"\n")

        f.write("Kind cars " + str(self.kind_cars) +"\n")
        f.write("Input length " + str(input_sequence_length) +"\n")
        f.write("Output length " + str(output_sequence_length) +"\n")
        f.write("Capacity Lines " + str(capacity_lines) +"\n")
        f.write("Num Lines " + str(num_lines) +"\n")
        if multi_agent:
            f.write("Multi Agent\n")
        if curricular:
            f.write("Curricular - Front and Back same settings")

        f.close()


    def linearize(self, states):
        return fc.linearize(states, self.kind_cars)

    def curricular_train(self, env, back_agent, front_agent, n_iterations, checkpoint_length, n_episodes=8000, eps_start=1.0, eps_end=0.0001, eps_decay=0.999, valid_actions_only = False):

        self.save_settings(env, back_agent, curricular = 'Back')
        self.save_settings(env, front_agent, curricular = 'Front')

        pathname = fc.get_path()

        ## Wrapping training of one learning agent against fixed wagent ##
        def train(learning_agent, wrapper, wrapper_first, best_score_weights_file, weights_file, scores_file, itr):

            agent_name = 'Back' if 'Back' in weights_file else 'Front'

            # if we train the back agent, a1 should be None, because back agent starts.
            ###  maw = MAW(env, None, wrapper)
            # if we train the front agent, a2 should be None, for the same reason
            ### maw = MAW(env, wrapper, None)
            a1, a2 = [wrapper, None][::wrapper_first]
            maw = MAW(env, a1, a2)

            # status variables and plots
            scores = []  # list containing scores from each episode
            means = []
            scores_window = deque(maxlen=checkpoint_length)  # last 100 scores
            eps = eps_start  # initialize epsilon

            # flush the replay buffer of previous training experiences
            learning_agent.reset_memory()

            
            for i_episode in range(1, n_episodes + 1):
                maw.reset()
                state = maw.get_state()
                state = self.linearize(state)
                score = 0
                done = False
                while not done:
                    action = learning_agent.act(state, eps, eval_mode = valid_actions_only, pa = env.possible_actions())
                    reward, next_state, done = maw.step(action)  # send the action to the environment and observe
                    next_state = self.linearize(next_state)
                    learning_agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)
                scores.append(score)
                eps = max(eps_end, eps_decay * eps)

                means.append(np.mean(scores_window))

                if i_episode % checkpoint_length == 0:

                    score = np.mean(scores_window)

                    # if current score is better, save the network weights and update best seen score
                    if score > self.best_score:
                        self.best_score = score
                        torch.save(learning_agent.qnetwork_local.state_dict(), best_score_weights_file)
                    print('\rCC Iteration {} ({}) Episode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}'.format(itr, agent_name, i_episode, score, self.best_score))

                    f = open(scores_file, "a")
                    for score in scores_window:
                        f.write(str(score) + '\n')
                    f.close()

            torch.save(learning_agent.qnetwork_local.state_dict(), weights_file)

        ## Perform Curricular Training ##

        # store scores of all iteration as list of tuples [(score_back_1,score_back_2), ..., (score_back_niter, score_front_niter)]
        all_scores = []

        # initially use a random front agent
        front_wrapper = fc.RandomPlayer()

        for iteration in range(1,n_iterations+1):

            # filenames for this iteration
            best_score_weights_filename_front = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + "_ITR:{itr}_Front_highScore.pth".format(itr = iteration)
            weights_filename_front = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_ITR:{itr}_Front.pth'.format(itr = iteration)

            best_score_weights_filename_back = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_ITR:{itr}_Back_highScore.pth'.format(itr = iteration)
            weights_filename_back = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_ITR:{itr}_Back.pth'.format(itr = iteration)

            scores_filename_back = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_ITR:{itr}_Back.scores'.format(itr = iteration)
            scores_filename_front = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_ITR:{itr}_Front.scores'.format(itr = iteration)

            # train back with fixed front
            scores_back = train(back_agent, front_wrapper, 1, best_score_weights_filename_back, weights_filename_back, scores_filename_back, iteration)
            back_wrapper = fc.Agent_wrapper(back_agent.qnetwork_local, env.kind_cars)

            # train front with fixed back
            scores_front = train(front_agent, back_wrapper, -1, best_score_weights_filename_front, weights_filename_front, scores_filename_front, iteration)
            front_wrapper = fc.Agent_wrapper(front_agent.qnetwork_local, env.kind_cars)

            all_scores.append((scores_back,scores_front))

        return all_scores
        
    def multi_train(self,env, agent_front, agent_back, checkpoint_length, n_episodes=8000, eps_start=1.0, eps_end=0.0001, eps_decay=0.999, show_picture = False, valid_actions_only = False):
        self.save_settings(env, agent_front, multi_agent= True)
        pathname = fc.get_path()

        best_score_weigths_filename_front = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + "_Front_highScore.pth"
        weights_filename_front = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_Front.pth'

        best_score_weigths_filename_back = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_Back_highScore.pth'
        weights_filename_back = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '_Back.pth'

        scores_filename = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '.scores'


        scores = []  # list containing scores from each episode
        means = []
        scores_window = deque(maxlen=checkpoint_length)  # last 100 scores
        eps = eps_start  # initialize epsilon

        # init figure
        fig = plt.figure()
        # ax = fig.add_subplot(111)

        for i_episode in range(1, n_episodes + 1):
            env.reset()
            state = env.get_state()
            score = 0
            done = False
            while not done:
                state = self.linearize(env.get_state())
                player = env.get_player()
                if player == 1:
                    action = agent_front.act(state, eps, eval_mode= valid_actions_only, pa = env.possible_actions())
                else:
                    action = agent_back.act(state, eps, eval_mode= valid_actions_only, pa = env.possible_actions())

                reward, next_state, done = env.step(action)  # send the action to the environment and observe
                next_state = self.linearize(next_state)

                if player == 1:
                    agent_front.step(state, action, reward, next_state, done)
                else:
                    agent_back.step(state, action, reward, next_state, done)

                score += reward
                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)

            means.append(np.mean(scores_window))

            if i_episode % checkpoint_length == 0:

                score = np.mean(scores_window)

                # if current score is better, save the network weights and update best seen score
                if score > self.best_score:
                    self.best_score = score
                    torch.save(agent_front.qnetwork_local.state_dict(), best_score_weigths_filename_front)
                    torch.save(agent_back.qnetwork_local.state_dict(), best_score_weigths_filename_back)
                if show_picture:
                    plt.scatter(range(len(scores)), scores, label='Scores', color='c', alpha=0.8, s=1)
                    plt.plot(np.arange(len(means)), means, label='Mean', color='r')
                    plt.ylabel('Score')
                    plt.xlabel('Episode #')
                    plt.show()
                print(
                    '\rEpisode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}'.format(i_episode, score, self.best_score))

                f = open(scores_filename, "a")
                for score in scores_window:
                    f.write(str(score) + '\n')
                f.close()

        torch.save(agent_front.qnetwork_local.state_dict(), weights_filename_front)
        torch.save(agent_back.qnetwork_local.state_dict(), weights_filename_back)
        return scores


    def train(self, env, agent, checkpoint_length, n_episodes=8000, eps_start=1.0, eps_end=0.0001,
                 eps_decay=0.999, show_picture = False, valid_actions_only = False):

        # initialise saving spots for better reading in the code
        run_number = self.save_settings(env, agent)
        pathname = fc.get_path()

        #best_score_weigths_filename = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name  + '_highScore.pth'
        #weights_filename = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name +  '.pth'
        #scores_filename = pathname + "/results/" + self.checkpoint_name + "/" + self.checkpoint_name + '.scores'
        
        best_score_weights_filename = "/media/shadowwalker/DATA/study/RIL1/code/carmanufacturing/test/results/CP_NoVal/bestscore_32.pth"
        weights_filename = "/media/shadowwalker/DATA/study/RIL1/code/carmanufacturing/test/results/CP_NoVal/weights_32.pth"
        scores_filename = "/media/shadowwalker/DATA/study/RIL1/code/carmanufacturing/test/results/CP_NoVal/scores.txt"


        # status variables and plots
        scores = []  # list containing scores from each episode
        means = []
        scores_window = deque(maxlen=checkpoint_length)  # last 100 scores
        eps = eps_start  # initialize epsilon

        for i_episode in range(1, n_episodes + 1):
            env.reset(i_episode)
            state = env.get_state()
            state = self.linearize(state)
            score = 0
            done = False
            while not done:
                action = agent.act(state, eps, eval_mode = valid_actions_only, pa = env.possible_actions())
                reward, next_state, done = env.step(action)  # send the action to the environment and observe
                next_state = self.linearize(next_state)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)

            means.append(np.mean(scores_window))

            if i_episode % checkpoint_length == 0:

                score = np.mean(scores_window)

                # if current score is better, save the network weights and update best seen score
                if score > self.best_score:
                    self.best_score = score
                    torch.save(agent.qnetwork_local.state_dict(), best_score_weights_filename)
                if show_picture:
                    clear_output(wait=True)
                    #plt.scatter(range(len(scores)), scores, label='Scores', color='c', alpha=0.8, s=1)
                    plt.plot(np.arange(len(means)), means, label='Mean', color='r')
                    plt.ylabel('Score')
                    plt.xlabel('Episode #')
                    plt.show()
                print('\rEpisode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}'.format(i_episode, score, self.best_score))

                f = open(scores_filename, "a")
                for score in scores_window:
                    f.write(str(score) + '\n')
                f.close()

        torch.save(agent.qnetwork_local.state_dict(), weights_filename)
        return scores


    def test_agent(self, env, agent, test_sequences, checkpoint_length, buffers = [], output_sequences = []):
        scores = []  # list containing scores from each episode
        means = []
        scores_window = deque(maxlen=100)
        actions = []
        total_counter = 0
        hits = []

        for i_episode, test_sequence in enumerate(test_sequences):
            no_hit = 0
            small_hit = 0
            big_hit = 0
            huge_hit = 0
            current_actions = []
            env.reset(test_sequence)
            score = 0
            done = False
            while not done:
                total_counter += 1
                action = agent.act(env, eval_mode = False)
                current_actions.append(action)
                reward, next_state, done = env.step(action)  # send the action to the environment and observe
                score += reward #* np.power(self.gamma, t)
                #if reward == -100:
                #    print("invalid, terminating")
                    # that was an invalid move, terminate game!
                #    break
                if reward == 0:
                    no_hit += 1
                else:
                    if reward == -3:
                        small_hit += 1
                    else:
                        if reward == -5:
                            big_hit += 1
                        else:
                            huge_hit += 1
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            means.append(np.mean(scores_window))
            actions.append(current_actions)
            hits.append([no_hit, small_hit, big_hit, huge_hit])

            if i_episode % checkpoint_length == - 1 % checkpoint_length:
                score = np.mean(scores_window)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score))

        return scores, means, actions, total_counter, hits

    def test_multi_agents(self, env, agent_front, agent_back, test_sequences, checkpoint_length):
        scores = []  # list containing scores from each episode
        means = []
        scores_window = deque(maxlen=checkpoint_length)
        actions = []
        total_counter = 0
        hits = []

        for i_episode, test_sequence in enumerate(test_sequences):
            no_hit = 0
            small_hit = 0
            big_hit = 0
            huge_hit = 0
            current_actions = []
            env.reset(test_sequence)
            score = 0
            done = False
            while not done:
                total_counter += 1
                if env.get_player() == 1:
                    action = agent_front.act(env, eval_mode = True)
                else:
                    action = agent_back.act(env, eval_mode = True)

                current_actions.append(action)
                reward, next_state, done = env.step(action)  # send the action to the environment and observe
                score += reward
                if reward == -100:
                    print("invalid, terminating! Player: ", env.get_player())
                    # that was an invalid move, terminate game!
                    break
                if reward == 0:
                    no_hit += 1
                else:
                    if reward == -3:
                        small_hit += 1
                    else:
                        if reward == -5:
                            big_hit += 1
                        else:
                            huge_hit += 1
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            means.append(np.mean(scores_window))
            actions.append(current_actions)
            hits.append([no_hit, small_hit, big_hit, huge_hit])


            if i_episode % checkpoint_length == - 1 % checkpoint_length:
                score = np.mean(scores_window)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score))

        return scores, means, actions, total_counter, hits

    def get_optimal_score(self, env, test_sequences, checkpoint_length, depth_search = 10):
        las = Las(depth_search)

        scores = []  # list containing scores from each episode
        means = []
        scores_window = deque(maxlen=checkpoint_length)
        actions = []
        hits = []
        total_counter = 0

        for i_episode, test_sequence in enumerate(test_sequences):
            current_actions = []
            no_hit = 0
            small_hit = 0
            big_hit = 0
            huge_hit = 0
            env.reset(test_sequence)
            score = 0
            done = False
            while not done:
                total_counter += 1

                action = las.act(env)

                current_actions.append(action)
                reward, next_state, done = env.step(action)  # send the action to the environment and observe
                score += reward
                if reward == -100:
                    # that was an invalid move, terminate game! (actually inpossible for optimal solver)
                    break
                if reward == 0:
                    no_hit += 1
                else:
                    if reward == -3:
                        small_hit += 1
                    else:
                        if reward == -5:
                            big_hit += 1
                        else:
                            huge_hit += 1
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            means.append(np.mean(scores_window))
            actions.append(current_actions)
            hits.append([no_hit, small_hit, big_hit, huge_hit])

            if i_episode % checkpoint_length == - 1 % checkpoint_length:
                score = np.mean(scores_window)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), "Search Depth:", depth_search)

        return scores, means, actions, total_counter, hits

