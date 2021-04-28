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
        'sp4_100k': "6 set size",
        'sp6_100k': "12 set size",
        'sp5_100k': "20 set size"
    }

    def sub_plot_r2(ax, key, arr):
        arr = arr[:, :200]

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
        print("Last Loss", key, mean[-1], "+-", mean[-1] - ci[0][-1])

    for k in tags:
        v = kv[k]
        sub_plot_r2(ax, k, v[1])  # This is train loss.
        ax.legend(frameon=False)
        ax.set_ylabel(r'Train Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylim([0, 30])
        # ax.set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('sp_loss.png')
    fig.savefig('sp_loss.pdf')


def plot_vars(kv):
    fig, ax = plt.subplots(figsize=(4, 3))

    tags = {
        'sp4_100k': "6 set size",
        'sp6_100k': "12 set size",
        'sp5_100k': "20 set size"
    }

    def sub_plot_r2(ax, key, arr):
        arr = arr[:, :200]

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
        print("Last R2", key, mean[-1], "+-", mean[-1] - ci[0][-1])

    for k in tags:
        v = kv[k]
        sub_plot_r2(ax, k, v[2])  # This is R**2
        ax.legend(frameon=False)
        ax.set_ylabel(r'$R^2$')
        ax.set_xlabel("Epoch")
        ax.set_ylim([0, .5])
        # ax.set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('sp_vars.png')
    fig.savefig('sp_vars.pdf')


def plot_sort(kv):
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.5))

    tags = {
        'sp4_100k': "6 set size",
        'sp6_100k': "12 set size",
        'sp5_100k': "20 set size"
    }
    tags_sort = {
        'sp4_sort_100k': "6 set size, sorted",
        'sp6_sort_100k': "12 set size, sorted",
        'sp5_sort_100k': "20 set size, sorted"
    }

    def sub_plot_r2(ax, key, arr, tags):
        arr = arr[:, :300]

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
        print("Last R2", key, mean[-1], "+-", mean[-1] - ci[0][-1])

    for i, (k1, k2) in enumerate(zip(tags.keys(), tags_sort.keys())):
        print(k1, k2)
        ax = axs[i]
        sub_plot_r2(ax, k1, kv[k1][2], tags)
        sub_plot_r2(ax, k2, kv[k2][2], tags_sort)
        ax.legend(frameon=False)
        ax.set_ylabel(r'$R^2$')
        ax.set_xlabel("Epoch")
        ax.set_ylim([0, .6])
        # ax.set_title(tags[k])

    plt.tight_layout()
    plt.show()
    fig.savefig('sp_sorted.png')
    fig.savefig('sp_sorted.pdf')


def run():
    with open('sp_vars.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot_loss(kv)
    plot_vars(kv)
    plot_sort(kv)


if __name__ == '__main__':
    run()
