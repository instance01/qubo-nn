import pickle
import collections

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


def gen_table(kv):
    def calc_ci(key, arr):
        arr = arr[~np.isnan(arr)]
        arr = arr[arr != 0.]
        mean = np.mean(arr, axis=0)
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        range_ = round(mean - ci[0], 4)
        mean = round(mean, 4)
        return mean, range_

    for k, v in kv.items():
        # arr = arr[:, 1:201]
        v = np.array(v)
        mean, range_ = calc_ci(k, v[0].max(axis=1))  # r2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


def plot_5x5(kv):
    tags = {
        'np19_LONG_r2_5x5': '5x5, 5000 hidden nodes',
        'np19_LONG_r2_5x5_empty': '5x5, no hidden nodes'
    }

    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))

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

    for k, v in kv.items():
        if k not in tags:
            continue
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :1000])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend()
    plt.ylim((0.5, 1.05))
    plt.tight_layout()
    plt.show()
    fig.savefig('np_comp_nodes5x5.png')
    fig.savefig('np_comp_nodes5x5.pdf')


def plot_64x64(kv):
    tags = {
        'np19_LONG_r2': '64x64, no hidden nodes',
        'np19_LONG_r2_64x64_2k': '64x64, 2k hidden nodes',
        'np19_LONG_r2_64x64_5k': '64x64, 5k hidden nodes'
    }

    fig, axs = plt.subplots(1, 1, figsize=(5, 3.5))

    def calc_ci(ax, key, arr):
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

    for k, v in kv.items():
        if k not in tags:
            continue
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :5000])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend()
    plt.ylim((0.9, 1.01))
    plt.tight_layout()
    plt.show()
    fig.savefig('np_comp_nodes64x64.png')
    fig.savefig('np_comp_nodes64x64.pdf')


def run():
    with open('np_comp.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot_5x5(kv)
    plot_64x64(kv)


if __name__ == '__main__':
    run()
