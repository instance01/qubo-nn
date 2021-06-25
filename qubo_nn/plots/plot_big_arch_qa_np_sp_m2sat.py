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
        'sp_gen1_big_arch': '64x64',
        'sp_gen2_big_arch': '64x64, 49x49',
        'sp_gen3_big_arch': '64x64, 49x49, 36x36',
        'sp_gen4_big_arch': '64x64, 49x49, 36x36, 25x25'
    },
    {
        'm2sat_8x8_10_gen1_big_arch': '16x16',
        'm2sat_8x8_10_gen2_big_arch': '16x16, 14x14',
        'm2sat_8x8_10_gen3_big_arch': '16x16, 14x14, 12x12',
        'm2sat_8x8_10_gen4_big_arch': '16x16, 14x14, 12x12, 10x10'
    },
    {
        'qa_N_144_norm3_generalized_50k_gen1_big_arch': '144x144',
        'qa_N_144_norm3_generalized_50k_gen2_big_arch': '144x144, 100x100',
        'qa_N_144_norm3_generalized_50k_gen3_big_arch': '144x144, 100x100, 64x64',  # noqa
        'qa_N_144_norm3_generalized_50k_gen4_big_arch': '144x144, 100x100, 64x64, 36x36'  # noqa
    },
    {
        'np19_LONG_generalized_gen1_big_arch': '64x64',
        'np19_LONG_generalized_gen2_big_arch': '64x64, 48x48',
        'np19_LONG_generalized_gen3_big_arch': '64x64, 48x48, 32x32',
        'np19_LONG_generalized_gen4_big_arch': '64x64, 48x48, 32x32, 24x24'
    }
]
PLOT_NAMES = [
    'sp', 'm2sat', 'qa', 'np'
]
PLOT_LIMS = [
    (0.0, 0.5, 100),
    (0.0, 1.0, 90),
    (0.95, 1.01, 2000),
    (0.95, 1.01, 10000)
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

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k in tags:
        v = kv[k]
        v = np.array(v)
        calc_ci(axs, k, v[0][:, :lims[-1]])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
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
    gen_table(kv)
    for plot_tags, plot_name, lims in zip(PLOT_TAGS, PLOT_NAMES, PLOT_LIMS):
        plot(kv, plot_tags, plot_name, lims)


if __name__ == '__main__':
    run()
