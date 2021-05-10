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
tags = {
    "gc1_r2": "GC",
    "tsp2_r2": "TSP",
    "a19_2_r2": "MC",
    "mvc3_r2": "MVC",
    "m2sat_16x16_5_F_v2": "M2SAT",
    "np19_LONG_r2": "NP",
    "qa_N_100_norm3": "QA",
    "sp4": "SP"
}


def plot_matrix(cfg_id, ax, axs):
    base_path = '../models/'
    model_fname = base_path + kv[cfg_id]
    with open(base_path + cfg_id + '.pickle', 'rb') as f:
        output_size = pickle.load(f)

    cfg = Config('../').get_cfg(cfg_id)
    optimizer = ReverseOptimizer(cfg, None, None, output_size)
    optimizer.load(model_fname, output_size)

    # for name, param in optimizer.net.named_parameters():
    #     print(name, param)

    # fig, ax = plt.subplots(figsize=(8, 8))
    mat = list(optimizer.net.named_parameters())[0][1].detach().numpy()
    size = min(mat.shape[0], 192)
    im = ax.matshow(mat[:size, :size])
    # ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(tags[cfg_id])
    return im


fig, axs = plt.subplots(2, 4, figsize=(8, 3), constrained_layout=True)
axs = [ax for sub_ax in axs for ax in sub_ax]
for k, ax in zip(kv, axs):
    im = plot_matrix(k, ax, axs)

cbar = ax.figure.colorbar(im, ax=[axs], aspect=50)
cbar.ax.set_ylabel('Weight Value', rotation=-90, va="bottom")

plt.savefig('weights_all.png')
plt.savefig('weights_all.pdf')
plt.show()
