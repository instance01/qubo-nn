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
        'sim_0_0': '0%/0%',
        'sim_1_1': '10%/10%',
        'sim_2_2': '20%/20%',
        'sim_3_3': '30%/30%',
        'sim_4_3': '40%/30%',
        'sim_4_4': '40%/40%',
        'sim_5_4': '50%/40%',
        'sim_6_3': '60%/30%',
        'sim_6_4': '60%/40%'
    }
]
PLOT_ACC = [
    {
        'sim_6_4': (0.311025, 0.1426625),
        'sim_6_3': (0.3453625, 0.1231125),
        'sim_5_4': (0.344175, 0.12655),
        'sim_4_4': (0.3433375, 0.12655),
        'sim_4_3': (0.2737875, 0.1273125),
        'sim_3_3': (0.2606875, 0.1273125),
        'sim_2_2': (0.2578, 0.123),
        'sim_1_1': (0.21785, 0.12255),
        'sim_0_0': (0., 0.)
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

    timeseries_acc_mc = []
    timeseries_acc_np = []
    timeseries_classif_mc = []
    timeseries_classif_mc_err = []
    for k in tags:
        timeseries_acc_mc.append(1 - PLOT_ACC[0][k][0])
        timeseries_acc_np.append(1 - PLOT_ACC[0][k][1])
        v = kv[k]
        v = np.array(v)
        x, mean, err = calc_ci(axs, k, v[0][:, :lims[-1]])  # r2
        timeseries_classif_mc.append(mean[-1])
        timeseries_classif_mc_err.append(err[-1])

    x = np.arange(len(timeseries_classif_mc))
    axs.errorbar(x, timeseries_classif_mc, timeseries_classif_mc_err, color=plt.cm.Set2.colors[2])  # noqa
    axs.set_xlabel('Noise factor (MC/NP)')
    ticks = list(tags.values())
    ticks.insert(0, '')
    axs.set_xticklabels(ticks)
    axs.set_ylabel('Misclassification rate', color=plt.cm.Set2.colors[2], fontsize=14, fontweight='bold')  # noqa

    axs2 = axs.twinx()
    axs2.plot(timeseries_acc_mc, color=plt.cm.Set2.colors[4], label='MC')
    axs2.plot(timeseries_acc_np, color=plt.cm.Set2.colors[4], label='NP')
    axs2.set_ylabel('QUBO Solution Quality', color=plt.cm.Set2.colors[4], fontsize=14, fontweight='bold')  # noqa
    labelLines(plt.gca().get_lines(), zorder=2.5)

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
