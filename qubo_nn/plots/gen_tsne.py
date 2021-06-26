import pickle
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from qubo_nn.data import LMDBDataLoader
from qubo_nn.config import Config


cfg_id = '27_scramble_10k_mini'
cfg = Config('../').get_cfg(cfg_id)
cfg["use_big"] = False
lmdb_loader = LMDBDataLoader(cfg, reverse=False, base_path='../')

X = []
y = []
for i, data in enumerate(lmdb_loader.train_data_loader):
    # if i > 100:
    if i > 20:
        break
    X.extend(data[0].tolist())
    y.extend(data[1].tolist())

X = np.array(X)
X = X.reshape(-1, 64**2)
print(X.shape)

for i in [10, 20, 30, 50, 70, 100, 200, 500, 1000]:
    tsne = TSNE(
        n_jobs=10,
        n_iter=5000,
        perplexity=i,
        # perplexity=500.,  # Best.
        verbose=1
    )
    Y = tsne.fit_transform(X)

    with open('tsne_data%d.pickle' % i, 'wb+') as f:
        pickle.dump((Y, y), f)
