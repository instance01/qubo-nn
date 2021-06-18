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
        if not v:
            continue
        if len(v[0]) == 0:
            continue
        v = np.array(v)
        mean, range_ = calc_ci(k, v[0].max(axis=1))  # r2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


PLOT_TAGS = [
    {
        'tsp2_r2': 'Old Model',
        'tsp_gen1': 'New Model'
    },
    {
        'tsp_gen1_81': 'QUBO size 81x81'
    }
]
PLOT_NAMES = ['comp', '81']


def plot(kv, tags, name):
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
        calc_ci(axs, k, v[0][:, :500])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    plt.legend(frameon=False)
    plt.ylim((0.95, 1.01))
    plt.tight_layout()
    plt.show()
    fig.savefig(NAME + '_' + name + '.png')
    fig.savefig(NAME + '_' + name + '.pdf')


def run():
    with open(NAME + '.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv, PLOT_TAGS[0], PLOT_NAMES[0])
    plot(kv, PLOT_TAGS[1], PLOT_NAMES[1])


if __name__ == '__main__':
    run()
