import random
import itertools
import numpy as np
from qubo_nn.problems.problem import Problem


class SetPartitioning(Problem):
    def __init__(self, cfg, set_, subsets, costs, P=10):
        self.set_ = set_
        self.subsets = subsets
        self.costs = costs
        self.P = P

    def gen_qubo_matrix(self):
        n = len(self.subsets)
        Q = np.zeros((n, n))

        for i in range(n):
            Q[i][i] += self.costs[i]

        for x in self.set_:
            curr_binary_rule = []
            for i, subset in enumerate(self.subsets):
                if x in subset:
                    curr_binary_rule.append(i)

            for i in curr_binary_rule:
                Q[i][i] -= self.P * 2

            tuples = itertools.product(curr_binary_rule, repeat=2)
            for j, k in tuples:
                if j == k:
                    Q[j][j] += self.P
                else:
                    Q[j][k] += self.P / 2.
                    Q[k][j] += self.P / 2.

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(20, 25), max_cost=10, **kwargs):
        problems = []
        set_ = list(range(size[0]))
        for _ in range(n_problems):
            subsets = []
            costs = []
            for _ in range(size[1]):
                x = list(filter(lambda x: random.random() < 0.5, set_))
                subsets.append(x)
                costs.append(int(random.random() * max_cost))
            problems.append((set_, subsets, costs))
        return [
            {"set_": set_, "subsets": subsets, "costs": costs}
            for (set_, subsets, costs) in problems
        ]
