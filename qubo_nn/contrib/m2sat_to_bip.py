import numpy as np
from qubo_nn.problems import Max2SAT


cfg = {
    "problems": {"M2SAT": {"check_duplicates": True}}
}


def create_BIP(qubo, orig_clauses):
    qubo_size = 8

    qubo[np.tril_indices(qubo_size, -1)] = 0.
    tmp = qubo.copy()
    tmp[np.tril_indices(qubo_size)] = 0.
    idx = np.where(tmp != 0)

    n_clauses = len(idx[0])

    BIP = np.zeros((qubo_size, n_clauses))
    bias = np.zeros((qubo_size,))
    y = np.zeros((n_clauses,))
    for i, clause in enumerate(orig_clauses):
        if clause[0][1] and clause[1][1]:
            y[i] = 1
        if not clause[0][1] and clause[1][1]:
            y[i] = 1

    clauses = list(zip(*idx))

    for i, clause in enumerate(clauses):
        val = qubo[clause[0]][clause[1]]

        if val > 0:
            BIP[clause[0]][i] = -1
            BIP[clause[1]][i] = -1
        elif val < 0:
            BIP[clause[0]][i] = +1
            BIP[clause[1]][i] = -1
            bias[clause[1]] -= 1

    # print(BIP)
    # print(bias)
    # print(np.diag(qubo))
    result = (BIP @ y) - bias
    print((result == np.diag(qubo)).all())


# UNITTEST:
qubo = np.array([
    [1., -0.5, 0., 0., 0., 0., 0.5, 0.],
    [0.,  1., -0.5, 0., -0.5, 0.5, 0., 0.],
    [0.,  0., 1., 0.5, 0., -0.5, 0., 0.],
    [0.,  0., 0., 0., 0., 0.5, -0.5, 0.],
    [0.,  0., 0., 0., 0., 0.5, -0.5, 0.],
    [0.,  0., 0., 0., 0., -2., 0., 0.],
    [0.,  0., 0., 0., 0., 0., 1., 0.],
    [0.,  0., 0., 0., 0., 0., 0., 0.]])
clauses = [[(0, False), (1, True)], [(0, False), (6, False)], [(1, False), (2, True)], [(1, False), (4, True)], [(1, True), (5, True)], [(2, False), (3, False)], [(2, False), (5, True)], [(3, False), (5, False)], [(3, True), (6, False)], [(4, False), (6, True)], [(4, True), (5, True)]]   # noqa
create_BIP(qubo, clauses)


problems = Max2SAT.gen_problems(cfg, 100, size=(8, 20))

for problem in problems:
    m = Max2SAT(cfg, **problem)
    qubo = m.gen_qubo_matrix()
    qubo[np.tril_indices(8, -1)] = 0.
    clauses = problem["clauses"]
    clauses = sorted(clauses, key=lambda x: (x[0][0], x[1][0]))
    create_BIP(qubo, clauses)
