import numpy as np
from qubo_nn.problems import SetPacking


cfg = {
    "problems": {"SP": {}}
}

# problems = SetPacking.gen_problems(cfg, 1, size=(5, 4))
# 
# for problem in problems:
#     m = SetPacking(cfg, **problem)
#     qubo = m.gen_qubo_matrix()
#     qubo[np.tril_indices(4, -1)] = 0.
#     print(problem)
#     print(qubo)


def gen(problem):
    m = SetPacking(cfg, **problem)
    qubo = m.gen_qubo_matrix()
    qubo[np.tril_indices(4, -1)] = 0.
    print(problem)
    print(qubo)


gen({'set_': [0, 1, 2, 3, 4], 'subsets': [[0, 1, 4], [0, 2, 3], [2, 3], [3, 4]]})
gen({'set_': [0, 1, 2, 3, 4], 'subsets': [[0, 1, 4], [1, 2, 3], [2, 3], [3, 4]]})
# gen({'set_': [0, 1, 2, 3, 4], 'subsets': [[0, 1], [1, 2, 3], [2, 3, 4], [2, 4]]})


# [0, 1]
# [0]
# [1, 2]
# [1, 2, 3]
# [0, 3]
# [[1. 1. 0. 0.]
#  [1. 0. 0. 0.]
#  [0. 1. 1. 0.]
#  [0. 1. 1. 1.]
#  [1. 0. 0. 1.]]
# {'set_': [0, 1, 2, 3, 4], 'subsets': [[0, 1, 4], [0, 2, 3], [2, 3], [3, 4]]}
# [[-2. -3.  0. -3.]
#  [ 0.  1. -6. -3.]
#  [ 0.  0.  1. -3.]
#  [ 0.  0.  0.  1.]]
