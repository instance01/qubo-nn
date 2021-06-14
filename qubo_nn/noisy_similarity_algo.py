import qubovert
import pyxis as px
import numpy as np
from scipy.spatial import distance

from qubo_nn.data import LMDBDataLoader


QUBO_SIZE = 8


def solve_qubo2(item):
    Q = qubovert.utils.matrix_to_qubo(item.reshape(QUBO_SIZE, QUBO_SIZE))
    sol = Q.solve_bruteforce(all_solutions=False)
    sol_ = [0 for _ in range(QUBO_SIZE)]
    for k, v in sol.items():
        sol_[k] = v
    return sol_


cfg = {
    'dataset_id': 'a3dbg',
    'use_big': False,
    'model': {
        'batch_size': 1,
        'train_eval_split': 1.0,
        'shuffle_data': False
    }
}

lmdb_loader = LMDBDataLoader(cfg)
loader = lmdb_loader.train_data_loader
data = list(loader)

np_data = data[0][0][0].numpy()
mc_data = data[1][0][0].numpy()
mvc_data = data[2][0][0].numpy()
sols0 = np.array([solve_qubo2(d) for d in np_data])
sols1 = np.array([solve_qubo2(d) for d in mc_data])

median0 = np.percentile(np_data, 50, axis=0)
median1 = np.percentile(mc_data, 50, axis=0)
median2 = np.percentile(mvc_data, 50, axis=0)


def make_similar(our_data, their_data, sols, factor=.1):
    cached_data = our_data.copy()

    diff = their_data - our_data
    our_data += diff * factor

    new_sol = np.array([solve_qubo2(d) for d in our_data])
    print(abs(new_sol - sols).sum())
    binary_err = 0
    real_err = 0
    total_val = 0
    for i, (new_sol_, sol) in enumerate(zip(new_sol, sols)):
        best_val = sol.T @ cached_data[i] @ sol
        new_val = new_sol_.T @ cached_data[i] @ new_sol_

        # print(sol)
        # print(new_sol_)
        # print("B", best_val)
        # print("N", new_val)
        # print('')

        binary_err += min(
            abs(new_sol_ - sol).sum(),
            abs((1 - new_sol_) - sol).sum()
        )
        real_err += abs(best_val - new_val)
        total_val += abs(best_val)
    return our_data, binary_err, real_err, total_val


factors = [.4, .4]

mc_data_ = mc_data.copy()
mc_data, binary_err_mc, real_err_mc, total_val_mc = make_similar(mc_data, np_data, sols1, factors[0])
np_data, binary_err_np, real_err_np, total_val_np = make_similar(np_data, mc_data_, sols0, factors[1])

total_len = QUBO_SIZE * len(np_data)
print("ERR MC", binary_err_mc, binary_err_mc / total_len, real_err_mc, real_err_mc / total_val_mc)
print("ERR NP", binary_err_np, binary_err_np / total_len, real_err_np, real_err_np / total_val_np)
print("COS DIST", distance.cosine(mc_data.flatten(), np_data.flatten()))


def save(data, cfg_id):
    labels = [0 for _ in range(int(len(data) / 2))]
    labels.extend([1 for _ in range(int(len(data) / 2))])
    labels = np.array(labels)
    dirpath = 'datasets/%s/'
    db = px.Writer(
        dirpath=dirpath % cfg_id,
        map_size_limit=60000,
        ram_gb_limit=60
    )
    db.put_samples('input', data, 'target', labels)
    db.close()


data = np.concatenate([mc_data, np_data])
cfg_id = "sim_%d_%d" % tuple(int(10 * f)  for f in factors)
save(data, cfg_id)
print(cfg_id)
