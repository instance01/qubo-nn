import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


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
        mean, range_ = calc_ci(k, v[0].min(axis=1))  # fpfn_tot_ratio (reverse acc)
        print(k, "ACC", "%.3f" % (1 - mean), "+-", "%.3f" % range_)
        mean, range_ = calc_ci(k, v[1].max(axis=1))  # r2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


def plot(kv):
    tags = {
        'm2sat_16x16_5_F_v2': '5 clauses',
        'm2sat_16x16_10_F_v2': '10 clauses',
        'm2sat_16x16_15_F_v2': '15 clauses',
        'm2sat_16x16_20_F_v2': '20 clauses',
        'm2sat_16x16_25_F_v2': '25 clauses',
        'm2sat_16x16_30_F_v2': '30 clauses'
    }

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

        ax.plot(x, mean, label=tags[key])
        ax.fill_between(x, ci[0], ci[1], alpha=.2)

    for k, v in kv.items():
        calc_ci(axs, k, v[1][:, :20])  # r2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

        # # Shrink current axis by 20%
        # box = axs.get_position()
        # axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    fig.savefig('m2sat_comp.png')
    fig.savefig('m2sat_comp.pdf')


def run():
    with open('m2sat_comp.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv)


if __name__ == '__main__':
    run()
