import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
n = 5
color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)

# TAGS = {
#     'm3sat_5V_10_A': '5 vars, 10 clauses, 1M',
#     'm3sat_10V_10_A': '10 vars, 10 clauses, 1M',
#     'm3sat_15V_10_A': '15 vars, 10 clauses, 1M'
# }
TAGS = {
    'm3sat_10_F': '10 vars, 10 clauses, 100k',
    'm3sat_15V_10_F': '15 vars, 10 clauses, 100k',
    'm3sat_20V_10_F': '20 vars, 10 clauses, 100k'
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


def plot(kv, tags1, id_):
    fig, axs = plt.subplots(1, 1, figsize=(6, 3.0))

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

    for i, k in enumerate(tags1.keys()):
        v = kv[k]
        calc_ci(axs, k, v[2][:, :100], tags1, color[i+1])  # R2

        axs.legend()
        axs.set_ylabel(r'$R^2$')
        axs.set_xlabel("Epoch")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig('m3sat_comp_A_F_vars_%s.png' % id_)
    fig.savefig('m3sat_comp_A_F_vars_%s.pdf' % id_)


def run():
    with open('m23sat.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv, TAGS, 'c10')


if __name__ == '__main__':
    run()
