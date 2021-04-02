import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt
import cycler


np.seterr(divide='ignore', invalid='ignore')

mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')

# Source: https://stackoverflow.com/questions/30079590/
# For continuous:
# n = 10
# color = plt.cm.viridis(np.linspace(0, 1, n))
# mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
# For discrete:
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


def plot(kv):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.0))
    axs = axs.flatten()

    tags = {
        'a19_2_r2': 'Maximum Cut',
        'np19_LONG_r2': 'Number Partitioning',
        'gc1_r2': 'Graph Coloring',
        'mvc3_r2': 'Minimum Vertex Cover',
        'tsp2_r2': ' Traveling Salesman'
        # 'a19_gen_edges': 'Maximum Cut - edges'
    }

    def sub_plot(ax, key, arr):
        arr = arr[:, :1000]
        next(ax._get_lines.prop_cycler)['color']

        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for i, (k, v) in enumerate(kv.items()):
        # sub_plot(k, v[0])  # This is eval
        sub_plot(axs[i], k, v[1])  # This is train
        # ax.legend()
        if i == 0:
            axs[i].set_ylabel("Train Loss")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylim([-.5, 10])
        axs[i].set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('reverse_loss.png')
    fig.savefig('reverse_loss.pdf')


def plot_r2(kv):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.0))
    axs = axs.flatten()

    tags = {
        'a19_2_r2': 'Maximum Cut',
        'np19_LONG_r2': 'Number Partitioning',
        'gc1_r2': 'Graph Coloring',
        'mvc3_r2': 'Minimum Vertex Cover',
        'tsp2_r2': ' Traveling Salesman'
        # 'a19_gen_edges': 'Maximum Cut - edges'
    }

    def sub_plot_r2(ax, key, arr):
        # arr = arr[:, :5000]

        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)
        print("R2", key, mean[-1], "+-", mean[-1] - ci[0][-1])

    for i, (k, v) in enumerate(kv.items()):
        # sub_plot(k, v[0])  # This is eval
        sub_plot_r2(axs[i], k, v[2])  # This is R**2
        # axs[i].legend()
        if i == 0:
            axs[i].set_ylabel(r'$R^2$')
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylim([0, 1.1])
        axs[i].set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('reverse_r2.png')
    fig.savefig('reverse_r2.pdf')


def run():
    with open('reverse_losses.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot(kv)
    plot_r2(kv)


if __name__ == '__main__':
    run()
