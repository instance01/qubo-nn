import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def plot(id_):
    with open('tot_misclassifications_%s.pickle' % id_, 'rb') as f:
        arr1 = pickle.load(f)

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

        ax.plot(x, mean, color=col)
        ax.fill_between(x, ci[0], ci[1], color=col, alpha=.1)

    sub_plot(arr1, 'c')
    plt.ylabel("Misclassification ratio")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()
    fig.savefig('tot_mc_%s.png' % id_)
    fig.savefig('tot_mc_%s.pdf' % id_)


def run():
    plot('27_scramble_100k')
    plot('18_lr2_leaky')
    plot('23')


if __name__ == '__main__':
    run()
