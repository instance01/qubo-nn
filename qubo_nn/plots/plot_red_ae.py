import os
import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt

from qubo_nn.plots.lib import cmap_mod


NAME = os.path.splitext(os.path.basename(__file__))[0][5:]

mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)

PLOT_NAMES = [
    "np", "mc", "mvc", "sp", "m2sat", "spp", "gc", "qa", "qk", "m3sat", "tsp",
    "gi", "sgi", "mcq"
]


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
        if not v:
            continue
        if len(v[0]) == 0:
            continue
        v = np.array(v)
        mean, range_ = calc_ci(k, v[0].max(axis=1))  # r2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


def plot(kv, name):
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    tags = {}
    for i in range(1, 20):
        tags["red_" + name + "_" + str(i)] = str(i * 5) + "%"

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
        calc_ci(axs, k, v[0][:, :400])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig(NAME + '_' + name + '.png')
    fig.savefig(NAME + '_' + name + '.pdf')


def _get_data(kv, name):
    ret = []

    tags = {}
    for i in range(1, 20):
        tags["red_" + name + "_" + str(i)] = str(i * 5) + "%"

    def calc_ci(key, arr):
        mean = np.mean(arr, axis=0)
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )
        return mean, ci

    for k in tags:
        v = kv[k]
        v = np.array(v)
        x, ci = calc_ci(k, v[0][:, :400])  # r2
        ret.append(x[-1])

    return ret


def plot_all(kv):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for plot_name in PLOT_NAMES:
        data = _get_data(kv, plot_name)[::-1]
        x = np.arange(len(data))
        ax.plot(x, data, label=plot_name.upper())

    ax.legend()
    ax.set_ylabel(r'$R^2$')
    ax.set_xlabel("Epoch")

    tags = []
    for i in range(1, 20):
        tags.append(str(i * 5) + "%")
    ticks = list(tags[::-1])
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.arange(len(ticks)))

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig(NAME + '_all.png')
    fig.savefig(NAME + '_all.pdf')


def plot_matrix(kv):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    all_data = []
    for plot_name in PLOT_NAMES:
        data = _get_data(kv, plot_name)[::-1]
        all_data.append(data)

    tags = []
    for i in range(1, 20):
        tags.append(str(i * 5) + "%")
    ticks = list(tags[::-1])
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.arange(len(ticks)))

    ax.set_yticklabels([p.upper() for p in PLOT_NAMES])
    ax.set_yticks(np.arange(len(PLOT_NAMES)))

    ax.set_xlabel('Hidden layer size as percent of input size')

    im = ax.imshow(all_data, vmin=0, vmax=1, cmap=cmap_mod)
    cbar = ax.figure.colorbar(im, ax=[ax], aspect=30)
    cbar.ax.set_ylabel(r'$R^2$', rotation=-90, va="bottom")
    for i in range(len(PLOT_NAMES)):
        for j in range(len(all_data[0])):
            txt = "%.2f" % round(all_data[i][j], 2)
            col = '#000000' if all_data[i][j] > .65 else '#ffffff'
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=7.5)  # noqa

    plt.show()
    fig.savefig(NAME + '_all_matrix.png', bbox_inches='tight')
    fig.savefig(NAME + '_all_matrix.pdf', bbox_inches='tight')


def run():
    with open(NAME + '.pickle', 'rb') as f:
        kv = pickle.load(f)
    # gen_table(kv)
    # for plot_name in PLOT_NAMES:
    #     plot(kv, plot_name)
    #     break

    # plot_all(kv)
    plot_matrix(kv)


if __name__ == '__main__':
    run()
