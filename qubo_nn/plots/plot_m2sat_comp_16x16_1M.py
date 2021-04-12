import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
n = 8
color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)


TAGS = {
    'm2sat_16x16_5_F_v2_1M': '5 clauses, 1M',
    'm2sat_16x16_10_F_v2_1M': '10 clauses, 1M',
    'm2sat_16x16_15_F_v2_1M': '15 clauses, 1M',
    'm2sat_16x16_20_F_v2_1M': '20 clauses, 1M',
    'm2sat_16x16_25_F_v2_1M': '25 clauses, 1M',
    'm2sat_16x16_30_F_v2_1M': '30 clauses, 1M'
}
TAGS2 = {
    'm2sat_16x16_5_F_v2': '5 clauses',
    'm2sat_16x16_10_F_v2': '10 clauses',
    'm2sat_16x16_15_F_v2': '15 clauses',
    'm2sat_16x16_20_F_v2': '20 clauses',
    'm2sat_16x16_25_F_v2': '25 clauses',
    'm2sat_16x16_30_F_v2': '30 clauses'
}


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

    for k in TAGS.keys():
        v = kv[k]
        mean, range_ = calc_ci(k, v[2].max(axis=1))  # R2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


def plot(kv, kv2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0))

    def calc_ci(ax, key, arr, tags, col=None):
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

        if col is not None:
            ax.plot(x, mean, label=tags[key], c=col)
            ax.fill_between(x, ci[0], ci[1], alpha=.2, color=col)
        else:
            ax.plot(x, mean, label=tags[key])
            ax.fill_between(x, ci[0], ci[1], alpha=.2)

    color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]

    for i, k in enumerate(TAGS.keys()):
        v = kv[k]
        calc_ci(axs[0], k, v[2][:, :25], TAGS, color[i+1])  # R2

        axs[0].legend()
        axs[0].set_ylabel(r'$R^2$')
        axs[0].set_xlabel("Epoch")

    color = plt.cm.Blues(np.linspace(.3, 1, n))[::-1]

    for i, k in enumerate(TAGS2.keys()):
        v = kv2[k]
        calc_ci(axs[1], k, v[1][:, :25], TAGS2, color[i+1])  # R2

        axs[1].legend()
        axs[1].set_ylabel(r'$R^2$')
        axs[1].set_xlabel("Epoch")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig('m2sat_comp_16x16_1M.png')
    fig.savefig('m2sat_comp_16x16_1M.pdf')


def run():
    with open('m23sat.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)

    with open('m2sat_comp.pickle', 'rb') as f:
        kv2 = pickle.load(f)
    plot(kv, kv2)


if __name__ == '__main__':
    run()
