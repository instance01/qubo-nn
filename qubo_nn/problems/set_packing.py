import random
import itertools
import numpy as np
from qubo_nn.problems.problem import Problem


class SetPacking(Problem):
    def __init__(self, cfg, subset_matrix, P=6):
        self.subset_matrix = subset_matrix
        self.P = P

    def gen_qubo_matrix(self):
        n = self.subset_matrix.shape[1]
        Q = np.zeros((n, n))

        for i in range(n):
            Q[i][i] += 1

        for row in self.subset_matrix:
            idx = list(zip(*np.where(row > 0)))
            tuples = itertools.combinations(idx, 2)
            for j, k in tuples:
                Q[j][k] -= self.P / 2.
                Q[k][j] -= self.P / 2.

            if len(idx) == 1:
                Q[idx[0]][idx[0]] -= self.P / 2.

        return Q

    @classmethod
    def gen_matrix(cls, set_, subsets):
        B = np.zeros((len(set_), len(subsets)))

        for m, x in enumerate(set_):
            for i, subset in enumerate(subsets):
                if x in subset:
                    B[m][i] = 1

        return B

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        sorting = cfg["problems"]["SP"].get("sorting", False)

        problems = []

        uniques = set()

        set_ = list(range(size[0]))
        for _ in range(n_problems * 3):
            subsets = set()
            for _ in range(size[1]):
                x = list(filter(lambda x: random.random() < 0.5, set_))
                if not x:
                    continue
                subsets.add(tuple(x))
            if len(subsets) != size[1]:
                continue
            subsets = sorted(list(subsets))
            if tuple(subsets) in uniques:
                continue
            uniques.add(tuple(subsets))

            B = SetPacking.gen_matrix(set_, subsets)

            # Sort it.
            if sorting:
                y = np.array([2 ** i for i in range(len(subsets))])
                z = B @ y
                idx = np.argsort(z)
                B = B[idx]

            problems.append(B)
            if len(problems) == n_problems:
                break

        print("SP generated problems:", len(problems))
        return [{"subset_matrix": matrix} for matrix in problems]
