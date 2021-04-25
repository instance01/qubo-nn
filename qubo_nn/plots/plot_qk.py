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


def plot(kv):
    tags = {
        'qk7_6x6_high10_1M': '6x6, high10',
        'qk7_6x6_high30_1M': '6x6, high30'
    }

    fig, axs = plt.subplots(1, 1, figsize=(4, 3.5))

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
        calc_ci(axs, k, v[0][:, :51])  # r2

        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

        # # Shrink current axis by 20%
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig('qk_high.png')
    fig.savefig('qk_high.pdf')


def plot2(kv):
    tags = {
        'qk10_norm': '16x16, high30, normed, 200k',
        'qk8_6x6_high10_1M': '16x16, high10, 1M',
        'qk8_16x16_high30_1M': '16x16, high30, 1M'
    }

    fig, axs = plt.subplots(1, 1, figsize=(4, 3.5))

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

    for k in tags:
        v = kv[k]
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :51])  # r2

        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.ylim((0, 1.05))
    plt.tight_layout()
    plt.show()
    fig.savefig('qk_norm.png')
    fig.savefig('qk_norm.pdf')


def run():
    with open('qk.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv)
    plot2(kv)


if __name__ == '__main__':
    run()
