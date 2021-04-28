import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


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
        'qa_N_16_norm3': "16x16 normal",
        'qa_N_16_norm3_goddamn': "16x16 modified"
    }

    def sub_plot(ax, key, arr):
        arr = arr[:, :100]

        # next(ax._get_lines.prop_cycler)['color']

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
        ax.legend(frameon=False)
        if i == 0:
            ax.set_ylabel("Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylim([-.5, 10])

    plt.tight_layout()
    plt.show()
    fig.savefig('qa_ones_loss.png')
    fig.savefig('qa_ones_loss.pdf')


def plot_r2(kv):
    fig, ax = plt.subplots(figsize=(4, 3))

    tags = {
        'qa_N_16_norm3': "16x16 normal",
        'qa_N_16_norm3_goddamn': "16x16 modified"
    }

    def sub_plot_r2(ax, key, arr):
        arr = arr[:, :100]

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
        sub_plot_r2(ax, k, v[2])  # This is R**2
        ax.legend(frameon=False)
        if i == 0:
            ax.set_ylabel(r'$R^2$')
        ax.set_xlabel("Epoch")
        ax.set_ylim([0, 1.1])
        # ax.set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('qa_ones_r2.png')
    fig.savefig('qa_ones_r2.pdf')


def run():
    with open('qa_ones.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot_loss(kv)
    plot_r2(kv)


if __name__ == '__main__':
    run()
