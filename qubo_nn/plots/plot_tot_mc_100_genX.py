import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as colors


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


cmap_mod = truncate_colormap('Greens', minval=.5, maxval=.99)


def plot_confusion(id_):
    with open('tot_mc_100_genX_%s.pickle' % id_, 'rb') as f:
        arr1, mc_tables = pickle.load(f)

    # mc_table = np.mean(mc_tables, axis=0)
    mean = np.mean(mc_tables, axis=0)
    mean = mean.reshape(14 * 14)
    var = st.sem(mc_tables, axis=0)
    var = var.reshape(14 * 14)

    ci = st.t.interval(
        0.95,
        len(mc_tables) - 1,
        loc=mean,
        scale=var
    )

    mean = mean.reshape((14, 14))
    ci = mean - ci[0].reshape((14, 14))

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(mean.shape[0]):
        for j in range(mean.shape[0]):
            if np.isnan(ci[i][j]):
                ci[i][j] = 0.
            ax.text(
                j,
                i,
                '%.02f \n± %.02f' % (mean[i][j], ci[i][j]),
                ha="center",
                va="center",
                color="w",
                fontsize=9
            )
            if i == j:
                print(i, mean[i][i], ci[i][i])

    problems = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "GC", "QA", "QK", "M3SAT", "TSP", "GI", "SGI", "MCQ"]  # noqa
    plt.xticks(list(range(len(problems))), problems)
    plt.yticks(list(range(len(problems))), problems)

    plt.tight_layout()
    ax.imshow(mean, cmap=cmap_mod, vmin=0, vmax=1)
    plt.savefig('confusion_100_genX_%s.png' % id_)
    plt.savefig('confusion_100_genX_%s.pdf' % id_)


def plot(id_):
    with open('tot_mc_100_genX_%s.pickle' % id_, 'rb') as f:
        arr1, confusion_matrix = pickle.load(f)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    def sub_plot(arr, col):
        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])

        ci = st.t.interval(
            0.95,
            len(arr) - 1,
            loc=np.mean(arr, axis=0),
            scale=st.sem(arr, axis=0)
        )

        ax.plot(x, mean, color=col, label='Neural Network')
        ax.fill_between(x, ci[0], ci[1], color=col, alpha=.1)

        print(mean[-1], "+-", mean[-1] - ci[0][-1])

    sub_plot(arr1, 'c')
    plt.plot([0.00013294232] * arr1.shape[1], label='Decision Tree')
    plt.ylim((-0.001, 0.025))
    # plt.axhline(y=0.00013294232)
    plt.ylabel("Misclassification rate")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('tot_mc_100_genX_%s.png' % id_)
    fig.savefig('tot_mc_100_genX_%s.pdf' % id_)


def run():
    plot_confusion('100_genX')
    plot('100_genX')
    plot('100_genX_2')


if __name__ == '__main__':
    run()
