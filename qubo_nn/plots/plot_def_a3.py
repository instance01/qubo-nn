import os
import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


NAME = os.path.splitext(os.path.basename(__file__))[0][5:]

mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)

PLOT_TAGS = [
    {
        'a3_1': 'AutoEncoder + Similarity Loss'
    },
    {
        'a3_3': 'AutoEncoder + Similarity Loss + QBSolv Loss'
    }
]
PLOT_NAMES = [
    'a3_1',
    'a3_3'
]
PLOT_LIMS = [
    (-0.05, 1.01, 50000),
    (9, 16, 50000),
]


def plot(kv, tags, name, lims):
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.75))

    def calc_ci(ax, key, arr):
        # arr = arr[~np.isnan(arr)]
        # arr = arr[arr != 0.]
        mean = np.mean(arr, axis=0)
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        x = np.arange(len(mean))

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k in tags:
        v = kv[k]
        v = np.array(v)
        calc_ci(axs, k, v[1][:, :lims[-1]])  # train loss

        axs.legend()
        axs.set_ylabel('Train Loss')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.ylim(lims[0:2])
    plt.tight_layout()
    plt.show()
    fig.savefig(NAME + '_' + name + '.png')
    fig.savefig(NAME + '_' + name + '.pdf')


def run():
    with open(NAME + '.pickle', 'rb') as f:
        kv = pickle.load(f)
    for plot_tags, plot_name, lims in zip(PLOT_TAGS, PLOT_NAMES, PLOT_LIMS):
        plot(kv, plot_tags, plot_name, lims)


if __name__ == '__main__':
    run()
