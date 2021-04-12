import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
n = 10
color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)


TAGS = {
    'm2sat_8x8_5_F_v2_1M': '5 clauses, 1M',
    'm2sat_8x8_10_F_v2_1M': '10 clauses, 1M',
    'm2sat_8x8_15_F_v2_1M': '15 clauses, 1M',
    'm2sat_8x8_20_F_v2_1M': '20 clauses, 1M',
    'm2sat_8x8_25_F_v2_1M': '25 clauses, 1M',
    'm2sat_8x8_30_F_v2_1M': '30 clauses, 1M'
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


def plot(kv):
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.0))

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

        ax.plot(x, mean, label=TAGS[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k in TAGS.keys():
        v = kv[k]
        calc_ci(axs, k, v[2][:, :50])  # R2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

        # # Shrink current axis by 20%
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig('m2sat_comp_16x16_1M.png')
    fig.savefig('m2sat_comp_16x16_1M.pdf')


def run():
    with open('m23sat.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv)


if __name__ == '__main__':
    run()
