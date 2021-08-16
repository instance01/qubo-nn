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
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors[:4][::-1])

PLOT_TAGS = [
    {
        'v_mvc_gen1': '64x64',
        'v_mvc_gen2': '64x64, 48x48',
        'v_mvc_gen3': '64x64, 48x48, 32x32',
        'v_mvc_gen4': '64x64, 48x48, 32x32, 24x24',
    },
    {
        'v_np_gen1': '64x64',
        'v_np_gen2': '64x64, 48x48',
        'v_np_gen3': '64x64, 48x48, 32x32',
        'v_np_gen4': '64x64, 48x48, 32x32, 24x24',
    },
    {
        'v_gc_gen1': '64x64',
        'v_gc_gen2': '64x64, 49x49',
        'v_gc_gen3': '64x64, 49x49, 36x36',
        'v_gc_gen4': '64x64, 49x49, 36x36, 25x25',
    },
    {
        'v_sp_gen1': '20x20',
        'v_sp_gen2': '20x20, 16x16',
        'v_sp_gen3': '20x20, 16x16, 12x12',
        'v_sp_gen4': '20x20, 16x16, 12x12, 8x8',
    }
]
PLOT_NAMES = [
    'mvc', 'np', 'gc', 'sp'
]
PLOT_LIMS = [
    (0.9, 1.01, 30),
    (0.9, 1.01, 200),
    (0.8, 1.01, 24),
    (0.0, 0.6, 25)
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


def plot(kv, tags, name, lims):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))

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

        ax.plot(x, mean, label=tags[key], zorder=1)
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k in tags:
        v = kv[k]
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :lims[-1]])  # r2

        # axs.legend(facecolor='white')
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    # plt.legend(frameon=False, prop={'size': 8})
    legend = plt.legend(facecolor='white', prop={'size': 8})
    # legend.set_zorder(10)
    legend.get_frame().set_linewidth(0.0)
    plt.ylim(lims[0:2])
    plt.tight_layout()
    plt.show()
    fig.savefig(NAME + '_' + name + '.png')
    fig.savefig(NAME + '_' + name + '.pdf')


def run():
    with open(NAME + '.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    for plot_tags, plot_name, lims in zip(PLOT_TAGS, PLOT_NAMES, PLOT_LIMS):
        plot(kv, plot_tags, plot_name, lims)


if __name__ == '__main__':
    run()
