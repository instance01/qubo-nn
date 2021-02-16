import random
import itertools
import numpy as np
from qubo_nn.problems.problem import Problem


class SetPacking(Problem):
    def __init__(self, set_, subsets, P=6):
        self.set_ = set_
        self.subsets = subsets
        self.P = P

    def gen_qubo_matrix(self):
        n = len(self.subsets)
        Q = np.zeros((n, n))

        for i in range(n):
            Q[i][i] += 1

        for x in self.set_:
            curr_binary_rule = []
            for i, subset in enumerate(self.subsets):
                if x in subset:
                    curr_binary_rule.append(i)

            tuples = itertools.combinations(curr_binary_rule, 2)
            for j, k in tuples:
                Q[j][k] -= self.P / 2.
                Q[k][j] -= self.P / 2.

        return Q

    @classmethod
    def gen_problems(self, n_problems, size=(20, 25), **kwargs):
        problems = []
        for _ in range(n_problems):
            set_ = list(range(size[0]))
            subsets = []
            for _ in range(size[1]):
                x = list(filter(lambda x: random.random() < 0.5, set_))
                subsets.append(x)
            problems.append((set_, subsets))
        return [
            {"set_": set_, "subsets": subsets}
            for (set_, subsets) in problems
        ]
