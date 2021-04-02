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


def plot_loss(kv):
    fig, ax = plt.subplots(figsize=(4, 3))

    tags = {
        'a19_2_r2': 'Adjacency matrix',
        'a19_2_r2_gen_edges2': 'Edge list'
    }

    def sub_plot(ax, key, arr):
        arr = arr[:, :500]

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
        sub_plot(ax, k, v[1])  # This is train
        ax.legend()
        if i == 0:
            ax.set_ylabel("Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylim([-.5, 10])
        # ax.set_title("Maximum Cut")

    plt.tight_layout()
    plt.show()
    fig.savefig('mc_comp_loss.png')
    fig.savefig('mc_comp_loss.pdf')


def plot_r2(kv):
    fig, ax = plt.subplots(figsize=(4, 3))

    tags = {
        'a19_2_r2': 'Adjacency matrix',
        'a19_2_r2_gen_edges2': 'Edge list'
    }

    def sub_plot_r2(i, ax, key, arr):
        arr = arr[:, :500]

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

        idx = np.argmax(mean)

        if i == 1:
            ax.add_artist(
                mpl.patches.Ellipse(
                    (idx, mean[idx]),
                    width=15.,
                    height=.05,
                    fc='none',
                    ec='r'
                )
            )

        print("Maximum R2", key, mean[idx], "+-", mean[idx] - ci[0][idx])

    for i, (k, v) in enumerate(kv.items()):
        # sub_plot(k, v[0])  # This is eval
        sub_plot_r2(i, ax, k, v[2])  # This is R**2
        ax.legend()
        if i == 0:
            ax.set_ylabel(r'$R^2$')
        ax.set_xlabel("Epoch")
        ax.set_ylim([0, 1.1])
        # ax.set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('mc_comp_edges_r2.png')
    fig.savefig('mc_comp_edges_r2.pdf')


def run():
    with open('mc_comp_edges.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot_loss(kv)
    plot_r2(kv)


if __name__ == '__main__':
    run()
