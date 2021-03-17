import os
import sys
import glob
import pickle
from collections import defaultdict

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cycler


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')

# Source: https://stackoverflow.com/questions/30079590/
# For continuous:
# n = 10
# color = plt.cm.viridis(np.linspace(0, 1, n))
# mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
# For discrete:
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot(kv):
    fig, ax = plt.subplots()

    tags = {
        'a19': 'Maximum Cut',
        'np19_LONG': 'Number Partitioning',
        'gc1': 'Graph Coloring',
        'mvc3': 'Minimum Vertex Cover',
        'a19_gen_edges': 'Maximum Cut - edges'
    }

    def sub_plot(key, arr):
        arr = arr[:, 1:1000]

        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        ci = 1.96 * np.std(arr, axis=0) / np.mean(arr, axis=0)

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, (mean - ci), (mean + ci), alpha=.2)

    for k, v in kv.items():
        # sub_plot(k, v[0])  # This is eval
        sub_plot(k, v[1])  # This is train

    ax.legend()
    plt.ylabel("NN Regression Loss")
    plt.xlabel("Epoch")
    plt.ylim([-.5, 10])
    plt.show()
    fig.savefig('reverse_loss.png')
    fig.savefig('reverse_loss.pdf')


def run():
    with open('reverse_losses.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot(kv)


if __name__ == '__main__':
    run()
