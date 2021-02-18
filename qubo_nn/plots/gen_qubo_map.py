import gzip
import pickle
import numpy as np


with gzip.open('../datasets/1.pickle.gz', 'rb') as f:
    data, labels = pickle.load(f)


# Calc percentiles.
singles = []
medians = []
for i in range(9):
    p = np.percentile(data[int(1e5 * i):int(1e5 * (i + 1))], 90, axis=0)
    medians.append(p)
    p = data[int(1e5 * i) + 100]
    singles.append(p)


with open('qubo_map.pickle', 'wb+') as f:
    pickle.dump((medians, singles), f)
