import pickle

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
        'qa_N_64_norm3_5p': '5',
        'qa_N_64_norm3_10p': '10',
        'qa_N_64_norm3_15p': '15',
        'qa_N_64_norm3_20p': '20',
        'qa_N_64_norm3_25p': '25',
        'qa_N_64_norm3_30p': '30',
        'qa_N_64_norm3_35p': '35',
        'qa_N_64_norm3_40p': '40',
        'qa_N_64_norm3_45p': '45',
        'qa_N_64_norm3_50p': '50',
        'qa_N_64_norm3_55p': '55',
        'qa_N_64_norm3_60p': '60',
        'qa_N_64_norm3_65p': '65',
        'qa_N_64_norm3_70p': '70',
        'qa_N_64_norm3_75p': '75',
        'qa_N_64_norm3_80p': '80',
        'qa_N_64_norm3_85p': '85',
        'qa_N_64_norm3_90p': '90'
    }

    def calc_ci(key, arr):
        # arr = arr[~np.isnan(arr)]
        # arr = arr[arr != 0.]
        mean = np.mean(arr, axis=0)
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        return mean, mean - ci[0]

    data = []
    err = []
    for k in tags:
        v = kv[k]
        v = np.array(v)
        mean, ci = calc_ci(k, v[0].max(axis=1))
        data.append(mean)
        err.append(ci)

    fig, axs = plt.subplots(1, 1, figsize=(6, 3.5))
    # axs.errorbar(np.arange(len(data)), data, yerr=err, fmt='-o')
    axs.errorbar(np.arange(len(data)), data, yerr=err, fmt='-')
    axs.set_ylabel(r'$R^2$')
    axs.set_xlim(-.2, 17.2)
    axs.set_ylim(0, 1.05)
    axs.set_xticks(np.arange(len(tags)))
    axs.set_xticklabels(list(tags.values()))
    axs.set_xlabel("Percentile removed")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    fig.savefig('qk_zeros.png')
    fig.savefig('qk_zeros.pdf')


def run():
    with open('qk_zeros.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    plot(kv)


if __name__ == '__main__':
    run()
