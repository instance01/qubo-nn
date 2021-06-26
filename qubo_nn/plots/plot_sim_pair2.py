import os
import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt
# from labellines import labelLines

from qubo_nn.plots.lib import cmap_mod


NAME = os.path.splitext(os.path.basename(__file__))[0][5:]

mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
# n = 6
# color = plt.cm.Greens(np.linspace(.3, 1, n))[::-1]
# mpl.rcParams['axes.prop_cycle'] = plt.cycler('color', color)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)

PLOT_TAGS = [
    {
        'sim_pair2_0_0': '0%/0%',
        'sim_pair2_5_5': '5%/5%',
        'sim_pair2_10_10': '10%/10%',
        'sim_pair2_20_20': '20%/20%',
        'sim_pair2_30_30': '30%/30%',
        'sim_pair2_40_30': '40%/30%',
        'sim_pair2_40_40': '40%/40%',
        'sim_pair2_50_40': '50%/40%',
        'sim_pair2_50_50': '50%/50%'
    }
]
PLOT_ACC = [
    {
        'sim_pair2_50_50': (0.3725888888888889, 0.29875555555555555, 0.30924444444444443, 0.35988888888888887, 0.1827888888888889, 0.29852222222222224, 0.24285555555555555, 0.0985111111111111, 0.28524444444444447, 0.29762222222222223, 0.1097111111111111),
        'sim_pair2_50_40': (0.2637111111111111, 0.15051111111111112, 0.12458888888888889, 0.36123333333333335, 0.0021777777777777776, 0.42156666666666665, 0.21152222222222222, 0.20206666666666667, 0.29941111111111113, 0.3038, 0.15123333333333333),
        'sim_pair2_40_40': (0.0, 0.4337, 0.3524111111111111, 0.29794444444444446, 0.24347777777777777, 0.21956666666666666, 0.18303333333333333, 0.259, 0.3559222222222222, 0.3523777777777778, 0.40805555555555556),
        'sim_pair2_40_30': (0.2578666666666667, 0.27166666666666667, 0.2591888888888889, 0.06537777777777778, 0.13322222222222221, 0.31201111111111113, 0.18768888888888888, 0.27355555555555555, 0.3674, 0.33923333333333333, 0.35544444444444445),
        'sim_pair2_30_30': (0.1834111111111111, 0.3353111111111111, 0.1581, 0.0007, 0.11857777777777778, 0.31201111111111113, 0.24482222222222222, 0.2911666666666667, 0.38406666666666667, 0.30445555555555553, 0.3511222222222222),
        'sim_pair2_20_20': (0.33226666666666665, 0.1723222222222222, 0.20721111111111112, 0.1463888888888889, 0.0, 0.0, 0.2715888888888889, 0.3047888888888889, 0.2054111111111111, 0.11474444444444444, 0.24335555555555555),
        'sim_pair2_10_10': (0.11877777777777777, 0.16175555555555557, 0.35057777777777777, 0.06362222222222222, 0.4169, 0.0, 0.0022555555555555554, 0.2995, 0.3517888888888889, 0.12215555555555556, 0.32784444444444444),
        'sim_pair2_5_5': (0.0377, 0.16923333333333335, 0.10471111111111112, 0.24316666666666667, 0.0003111111111111111, 0.29334444444444446, 0.1174, 0.14702222222222222, 0.0, 0.3558, 0.2726),
        'sim_pair2_0_0': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
    }
]
PLOT_NAMES = ['sim2']
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
        mean, range_ = calc_ci(k, v[0].min(axis=1))  # r2
        print(k, "R2", "%.3f" % mean, "+-", "%.3f" % range_)


def _get_data(kv, tags, lims):
    def calc_ci(key, arr):
        mean = np.mean(arr, axis=0)
        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )
        x = np.arange(len(mean))
        return x, mean, mean - ci[0]

    timeseries_acc_ = [[] for _ in range(11)]
    timeseries_classif = []
    timeseries_classif_err = []
    for k in tags:
        for j, acc in enumerate(PLOT_ACC[0][k]):
            timeseries_acc_[j].append(1 - acc)
        v = kv[k]
        v = np.array(v)
        x, mean, err = calc_ci(k, v[0][:, :lims[-1]])  # r2
        timeseries_classif.append(mean[-1])
        timeseries_classif_err.append(err[-1])

    timeseries_acc_ = np.array(timeseries_acc_)
    _, timeseries_acc, timeseries_acc_err = calc_ci('', timeseries_acc_)

    return timeseries_acc_, timeseries_acc, timeseries_acc_err, timeseries_classif, timeseries_classif_err  # noqa


def plot(kv, tags, name, lims):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    _, timeseries_acc, timeseries_acc_err, timeseries_classif, timeseries_classif_err = _get_data(kv, tags, lims)  # noqa

    x = np.arange(len(timeseries_classif))
    axs.errorbar(x, timeseries_classif, timeseries_classif_err, color=plt.cm.Set2.colors[2])  # noqa
    axs.set_xlabel('Noise factor between pair')
    ticks = list(tags.values())
    ticks.insert(0, '')
    axs.set_xticklabels(ticks)
    axs.set_ylabel('Misclassification rate', color=plt.cm.Set2.colors[2], fontsize=14, fontweight='bold')  # noqa

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


def plot_matrix(kv, tags, lims):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8.5))

    all_data, _, _, _, _ = _get_data(kv, tags, lims)  # noqa

    ticks = list(tags.values())
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_xticks(np.arange(len(ticks)))

    prob = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "QA", "QK", "M3SAT",
            "TSP", "MCQ"]
    ax.set_yticklabels(prob)
    ax.set_yticks(np.arange(len(prob)))

    ax.set_xlabel('Noise factor between pair')

    im = ax.imshow(all_data, vmin=0, vmax=1, cmap=cmap_mod)
    for i in range(len(prob)):
        for j in range(len(all_data[0])):
            txt = "%.2f" % round(all_data[i][j], 3)
            ax.text(j, i, txt, ha="center", va="center", color="#000000")
    cbar = ax.figure.colorbar(im, ax=[ax], aspect=30)
    cbar.ax.set_ylabel('Solution quality', rotation=-90, va="bottom")
    for i in range(5):
        plt.axhline(y=i * 2 + 1.5, c='#000000', lw=.5)
    plt.show()
    fig.savefig(NAME + '_all_matrix.png')
    fig.savefig(NAME + '_all_matrix.pdf')


def run():
    with open(NAME + '.pickle', 'rb') as f:
        kv = pickle.load(f)
    gen_table(kv)
    for plot_tags, plot_name, lims in zip(PLOT_TAGS, PLOT_NAMES, PLOT_LIMS):
        plot(kv, plot_tags, plot_name, lims)

    # TODO NOTE: PROBLEM ORDER IS DIFFERENT!!
    # TODO NOTE: PROBLEM ORDER IS DIFFERENT!!
    # TODO NOTE: PROBLEM ORDER IS DIFFERENT!!
    # plot_matrix(kv, PLOT_TAGS[0], PLOT_LIMS[0])


if __name__ == '__main__':
    run()
