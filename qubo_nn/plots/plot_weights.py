import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from qubo_nn.config import Config
from qubo_nn.pipeline import ReverseOptimizer


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


kv = {
    "gc1_r2": "21-04-01_13:16:25-2518151-indigiolith.cip.ifi.lmu.de-gc1_r2",
    "tsp2_r2": "21-04-01_13:35:04-9586534-danburit.cip.ifi.lmu.de-tsp2_r2",
    "a19_2_r2": "21-04-01_18:57:45-4208204-datolith.cip.ifi.lmu.de-a19_2_r2",
    "mvc3_r2": "21-04-02_01:17:28-4255422-feueropal.cip.ifi.lmu.de-mvc3_r2",
    "m2sat_16x16_5_F_v2": "21-04-02_07:40:21-3067619-feueropal.cip.ifi.lmu.de-m2sat_16x16_5_F_v2",
    "np19_LONG_r2": "21-04-03_03:56:39-5891150-indigiolith.cip.ifi.lmu.de-np19_LONG_r2",
    "qa_N_100_norm3": "21-04-06_21:10:40-7356487-beryll.cip.ifi.lmu.de-qa_N_100_norm3",
    "sp4": "21-04-09_12:45:49-1875734-feuerachat.cip.ifi.lmu.de-sp4"
}


def plot_matrix(cfg_id):
    base_path = '../models/'
    model_fname = base_path + kv[cfg_id]
    with open(base_path + cfg_id + '.pickle', 'rb') as f:
        output_size = pickle.load(f)

    cfg = Config('../').get_cfg(cfg_id)
    optimizer = ReverseOptimizer(cfg, None, None, output_size)
    optimizer.load(model_fname, output_size)

    # for name, param in optimizer.net.named_parameters():
    #     print(name, param)

    fig, ax = plt.subplots(figsize=(8, 8))
    mat = list(optimizer.net.named_parameters())[0][1].detach().numpy()
    size = min(mat.shape[0], 192)
    im = ax.matshow(mat[:size, :size])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    cbar = ax.figure.colorbar(im, ax=[ax], aspect=40)
    cbar.ax.set_ylabel('Avg Number of Stuns', rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig('weights_' + cfg_id + '.png')
    plt.savefig('weights_' + cfg_id + '.pdf')
    plt.show()
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.matshow(
    #     list(optimizer.net.named_parameters())[2][1].detach().numpy()[:256, :256],
    #     interpolation='none',
    #     filterrad=25000.
    # )
    # plt.show()


def plot_hist(cfg_id, labels=True):
    base_path = '../models/'
    model_fname = base_path + kv[cfg_id]
    with open(base_path + cfg_id + '.pickle', 'rb') as f:
        output_size = pickle.load(f)

    cfg = Config('../').get_cfg(cfg_id)
    optimizer = ReverseOptimizer(cfg, None, None, output_size)
    optimizer.load(model_fname, output_size)

    fig, ax = plt.subplots(figsize=(6, 6))

    # data = list(optimizer.net.named_parameters())[0][1].detach().numpy().flatten()
    # ax.hist(data, 30000)

    bins = 10000

    data = list(optimizer.net.named_parameters())[0][1].detach().numpy().flatten()
    ax.hist(data, bins)

    if cfg_id == 'a19_2_r2':
        axins = ax.inset_axes([0.1, 0.6, 0.3, 0.3])
        axins.hist(data[np.where(data < -.4)], bins)
        axins.set_xlim(-.51, -.45)
        axins.set_ylim(0, 20)
        if not labels:
            axins.get_xaxis().set_visible(False)
            axins.get_yaxis().set_visible(False)
        ax.indicate_inset_zoom(axins)
    elif cfg_id == 'mvc3_r2':
        axins = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
        axins.hist(data[np.where(data > .17)], bins)
        axins.set_xlim(.17, .23)
        axins.set_ylim(0, 20)
        if not labels:
            axins.get_xaxis().set_visible(False)
            axins.get_yaxis().set_visible(False)
        ax.indicate_inset_zoom(axins)

    if labels:
        plt.ylabel("Number of weights")
        plt.xlabel("Weight value")
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    if labels:
        plt.savefig('weights_hist_' + cfg_id + '2.png')
        plt.savefig('weights_hist_' + cfg_id + '2.pdf')
    else:
        plt.savefig('weights_hist_' + cfg_id + '.png')
        plt.savefig('weights_hist_' + cfg_id + '.pdf')
    plt.show()


for k in kv:
    plot_matrix(k)
    # plot_hist(k, False)
