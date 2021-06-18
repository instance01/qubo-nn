import os
import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt
from labellines import labelLines


NAME = os.path.splitext(os.path.basename(__file__))[0][5:]

mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)

PLOT_TAGS = [
    {
        'sim_pair_0_0': '0%/0%',
        'sim_pair_5_5': '5%/5%',
        'sim_pair_10_10': '10%/10%',
        'sim_pair_20_20': '20%/20%',
        'sim_pair_30_30': '30%/30%',
        'sim_pair_40_30': '40%/30%',
        'sim_pair_40_40': '40%/40%',
        'sim_pair_50_40': '50%/40%',
        'sim_pair_50_50': '50%/50%'
    }
]
PLOT_ACC = [
    {
        'sim_pair_50_50': (0.05782222222222222, 0.2882111111111111, 0.35863333333333336, 0.2479111111111111, 0.26572222222222225, 0.3688111111111111, 0.4051888888888889, 0.2576111111111111, 0.2932111111111111, 0.31345555555555554, 0.4158222222222222,),
        'sim_pair_50_40': (0.23114444444444446, 0.20703333333333335, 0.29762222222222223, 0.28524444444444447, 0.14295555555555556, 0.1743888888888889, 0.11662222222222222, 0.37835555555555556, 0.336, 0.1444111111111111, 0.3173,),
        'sim_pair_40_40': (0.0, 0.4337, 0.21152222222222222, 0.20206666666666667, 0.21593333333333334, 0.28025555555555554, 0.1827, 0.32284444444444443, 0.3587888888888889, 0.1308888888888889, 0.23842222222222223,),
        'sim_pair_40_30': (0.21593333333333334, 0.28025555555555554, 0.3569333333333333, 0.21482222222222222, 0.06593333333333333, 0.2139, 0.17184444444444444, 0.2466888888888889, 0.4337, 0.0, 0.32045555555555555,),
        'sim_pair_30_30': (0.4152888888888889, 0.0002555555555555556, 0.03292222222222222, 0.24663333333333334, 0.3062888888888889, 0.3097888888888889, 0.13557777777777777, 0.2529777777777778, 0.23454444444444444, 0.21875555555555556, 0.23516666666666666,),
        'sim_pair_20_20': (0.19664444444444446, 0.28473333333333334, 0.3292333333333333, 0.18611111111111112, 0.1282111111111111, 0.3569222222222222, 0.2446111111111111, 0.3198, 0.2785, 0.29812222222222223, 0.011788888888888889,),
        'sim_pair_10_10': (0.1297, 0.11577777777777777, 0.0, 0.2990888888888889, 0.2701777777777778, 0.1869888888888889, 0.35481111111111113, 0.21042222222222223, 0.35746666666666665, 0.0, 0.3307777777777778,),
        'sim_pair_5_5': (0.0, 0.3532777777777778, 0.3560333333333333, 0.0, 0.07845555555555556, 0.2401888888888889, 0.16111111111111112, 0.11887777777777778, 0.045144444444444445, 0.19444444444444445, 0.14235555555555557,),
        'sim_pair_0_0': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
    }
]
PLOT_NAMES = ['sim']
PLOT_LIMS = [(0.0, 1.0, 10000)]


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
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

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
        return x, mean, mean - ci[0]

        # ax.plot(x, mean, label=tags[key])
        # ax.fill_between(x, ci[0], ci[1], alpha=.2)

    timeseries_acc_ = [[] for _ in range(11)]
    timeseries_classif = []
    timeseries_classif_err = []
    for k in tags:
        for j, acc in enumerate(PLOT_ACC[0][k]):
            timeseries_acc_[j].append(1 - acc)
        v = kv[k]
        v = np.array(v)
        x, mean, err = calc_ci(axs, k, v[0][:, :lims[-1]])  # r2
        timeseries_classif.append(mean[-1])
        timeseries_classif_err.append(err[-1])

    timeseries_acc_ = np.array(timeseries_acc_)
    _, timeseries_acc, timeseries_acc_err = calc_ci(None, '', timeseries_acc_)

    x = np.arange(len(timeseries_classif))
    axs.errorbar(x, timeseries_classif, timeseries_classif_err, color=plt.cm.Set2.colors[2])  # noqa
    axs.set_xlabel('Noise between pair')
    ticks = list(tags.values())
    ticks.insert(0, '')
    axs.set_xticklabels(ticks)
    axs.set_ylabel('Misclassification rate', color=plt.cm.Set2.colors[2], fontsize=14, fontweight='bold')  # noqa

    prob = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "QA", "QK", "M3SAT",
            "TSP", "MCQ"]
    axs2 = axs.twinx()
    # print(timeseries_acc_err[0])
    axs2.errorbar(x, timeseries_acc, timeseries_acc_err, color=plt.cm.Set2.colors[4])  # noqa
    # for i, timeseries in enumerate(timeseries_acc):
        # axs2.plot(timeseries, color=plt.cm.Set2.colors[4], label=prob[i])
    axs2.set_ylabel('QUBO Solution Quality', color=plt.cm.Set2.colors[4], fontsize=14, fontweight='bold')  # noqa
    # labelLines(plt.gca().get_lines(), zorder=2.5)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.xticks(rotation=90)
    # plt.legend(frameon=False)
    # plt.ylim(lims[0:2])
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
