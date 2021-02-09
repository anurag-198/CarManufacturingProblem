import sys
import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class Plotter:

    def plot_training_curve(self, filename, slider_size = 100):
        pathname = os.path.dirname(sys.argv[0])
        if pathname == '':
            pathname = '.'

        pathname = pathname + "/results/" + filename + "/"

        for filename in os.listdir(pathname):
            if filename[-7:] != ".scores":
                continue

            scores  = np.loadtxt(pathname + filename)
            means = np.zeros(len(scores))
            scores_window = deque(maxlen=slider_size)

            for i,score in enumerate(scores):
                scores_window.append(score)
                means[i] = np.mean(scores_window)
            fig = plt.figure()


            plt.scatter(range(len(scores)), scores, label='Scores', color='c', alpha=0.8, s=1)
            plt.plot(np.arange(len(means)), means, label='Mean', color='r')
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()

    def plot_means(self, multiple_scores, labels, slider_size = 100):
        fig = plt.figure()
        for scores, label in zip(multiple_scores, labels):
            means = np.zeros(len(scores))
            scores_window = deque(maxlen=slider_size)
            for i, score in enumerate(scores):
                scores_window.append(score)
                means[i] = np.mean(scores_window)

            plt.plot(np.arange(len(means)), means, label=label)
        plt.show()