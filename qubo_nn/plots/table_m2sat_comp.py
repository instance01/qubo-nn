import os
import sys
import glob
import pickle
from collections import defaultdict

import scipy.stats as st
import numpy as np


def plot(kv):
    tags = {
        'm2sat_16x16_5_F_v2': 'Maximum Cut',
        'm2sat_16x16_5_F_v2': 'Number Partitioning',
        'm2sat_16x16_5_F_v2': 'Graph Coloring',
        'm2sat_16x16_5_F_v2': 'Minimum Vertex Cover',
        'm2sat_16x16_5_F_v2': 'Maximum Cut - edges',
        'm2sat_16x16_5_F_v2': 'Maximum Cut - edges'
    }

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


def run():
    with open('m2sat_comp.pickle', 'rb') as f:
        kv = pickle.load(f)
    plot(kv)


if __name__ == '__main__':
    run()
