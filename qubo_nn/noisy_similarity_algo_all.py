import qubovert
import pyxis as px
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from qubo_nn.data import LMDBDataLoader


QUBO_SIZE = 9


def solve_qubo2(item):
    Q = qubovert.utils.matrix_to_qubo(item.reshape(QUBO_SIZE, QUBO_SIZE))
    sol = Q.solve_bruteforce(all_solutions=False)
    sol_ = [0 for _ in range(QUBO_SIZE)]
    for k, v in sol.items():
        sol_[k] = v
    return sol_


cfg = {
    'dataset_id': 'a3dbg_all',
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

labels = [d[1][0].numpy() for d in data]
data = [d[0][0].numpy() for d in data]
data_copy = [d.copy() for d in data]

for x in data:
    print(np.percentile(np.diagonal(x, 0, 1, 2), 50, axis=0))

sols = [np.array([solve_qubo2(d) for d in curr_data]) for curr_data in data]
prob = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "QA", "QK", "M3SAT", "TSP", "MCQ"]


def make_similar(sols_all, factors=.1):
    total_len = QUBO_SIZE * len(data[0])
    for i in range(11):
        print("Making", i, "similar.")
        sols = sols_all[i]
        our_data = data[i]
        cached_data = our_data.copy()
        total_diff = None
        for j in range(11):
            if i == j:
                continue

            their_data = data_copy[j]
            if total_diff is None:
                total_diff = (their_data - our_data) * (1/11.)
            else:
                total_diff += (their_data - our_data) * (1/11.)
        our_data += total_diff * factors[i]

        new_sol = np.array([solve_qubo2(d) for d in our_data])
        print(abs(new_sol - sols).sum())
        binary_err = 0
        real_err = 0
        total_val = 0
        for k, (new_sol_, sol) in enumerate(zip(new_sol, sols)):
            best_val = sol.T @ cached_data[k] @ sol
            new_val = new_sol_.T @ cached_data[k] @ new_sol_

            binary_err += min(
                abs(new_sol_ - sol).sum(),
                abs((1 - new_sol_) - sol).sum()
            )
            real_err += abs(best_val - new_val)
            total_val += abs(best_val)
        print(prob[i], binary_err, binary_err / total_len, real_err, real_err / total_val)


factors = [0.6] + [.6] * 10
make_similar(sols, factors)

data_flattened = [d.flatten() for d in data]
print("COS DIST")
print(cosine_distances(data_flattened).round(2))


def save(data, labels, cfg_id):
    data_new = []
    for d in data:
        data_new.extend(d)
    data = np.array(data_new)
    labels_new = []
    for label in labels:
        labels_new.extend(label)
    labels = np.array(labels_new)
    dirpath = 'datasets/%s/'
    db = px.Writer(
        dirpath=dirpath % cfg_id,
        map_size_limit=60000,
        ram_gb_limit=60
    )
    db.put_samples('input', data, 'target', labels)
    db.close()


cfg_id = "sim_all_%d_%d" % tuple(int(100 * f)  for f in factors[:2])
save(data, labels, cfg_id)
print(cfg_id)
