import random
import itertools
import numpy as np
from qubo_nn.problems.problem import Problem


class ExactCover(Problem):
    def __init__(self, cfg, subset_matrix, A=2, B=2):
        self.subset_matrix = subset_matrix
        self.A = A
        self.B = B

    def gen_qubo_matrix(self):
        n = self.subset_matrix.shape[1]
        Q = np.zeros((n, n))

        for i in range(n):
            Q[i][i] -= self.A

        # From Lucas 2014: The second term, to find the minimum exact cover:
        for i in range(n):
            Q[i][i] += self.B

        for row in self.subset_matrix:
            idx = list(zip(*np.where(row > 0)))
            tuples = itertools.combinations(idx, 2)
            for j, k in tuples:
                Q[j][k] += self.A / 2
                Q[k][j] += self.A / 2

            if len(idx) == 1:
                Q[idx[0]][idx[0]] -= self.A

        return Q

    # TODO: This is 100% copy&paste from SetPacking... Refactor!!
    @classmethod
    def gen_matrix(cls, set_, subsets):
        B = np.zeros((len(set_), len(subsets)))

        for m, x in enumerate(set_):
            for i, subset in enumerate(subsets):
                if x in subset:
                    B[m][i] = 1

        return B

    # TODO: This is 100% copy&paste from SetPacking... Refactor!!
    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        sorting = cfg["problems"]["EC"].get("sorting", False)

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

            B = ExactCover.gen_matrix(set_, subsets)

            # Sort it.
            if sorting:
                y = np.array([2 ** i for i in range(len(subsets))])
                z = B @ y
                idx = np.argsort(z)
                B = B[idx]

            problems.append(B)
            if len(problems) == n_problems:
                break

        print("EC generated problems:", len(problems))
        return [{"subset_matrix": matrix} for matrix in problems]
