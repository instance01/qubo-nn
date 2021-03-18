import random
import itertools

import numpy as np
from qubo_nn.problems.problem import Problem


class Max3SAT(Problem):
    def __init__(self, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars

    def gen_qubo_matrix(self):
        n = self.n_vars + len(self.clauses)
        Q = np.zeros((n, n))

        # Clauses is a list of 3-tuples of 2-tuples.
        # Eg: [((0, True), (2, True), (3, False)), ((1, True), (3, False), (4, True))]
        # So we loop through all tuples and find the type of it.
        # There are 4 types: (True, True), (True, False), etc.
        for i, clause in enumerate(self.clauses):
            clause_idx = self.n_vars + i
            Q[clause_idx][clause_idx] += 2

            # (1 + w1) * (x1 + x2 + x3) = x1 + x2 + x3 + w1x1 + w1x2 + w1x3
            # (1 + w1) * ((1 - x1) + x2 + x3) = -x1 + x2 + x3 + w1 -w1x1 +w1x2 +w1x3
            # (1 + w1) * ((1 - x1) + x2 + (1 - x3)) = -x1 + x2 -x3 + 2w1 -w1x1 +w1x2 -w1x3
            for item in clause:
                item_idx = item[0]
                val = item[1]
                if val:
                    Q[item_idx][item_idx] -= 1
                    Q[clause_idx][item_idx] -= .5
                    Q[item_idx][clause_idx] -= .5
                if not val:
                    Q[item_idx][item_idx] += 1
                    Q[clause_idx][clause_idx] -= 1
                    Q[clause_idx][item_idx] += .5
                    Q[item_idx][clause_idx] += .5

            # -x1x2 -x1x3 -x2x3
            # -(1-x1)x2 -x1x3 -x2x3 = -1 +x1x2
            # -(1-x1)(1-x2) -x1x3 -x2x3 = -1 -2x1x2 +x1 +x2
            for (item1, item2) in itertools.combinations(clause, 2):
                idx1 = item1[0]
                idx2 = item2[0]
                val1 = item1[1]
                val2 = item2[1]
                if val1 and val2:
                    Q[idx1][idx2] += .5
                    Q[idx2][idx1] += .5
                if not val1 and val2:
                    Q[idx2][idx2] += 1.
                    Q[idx1][idx2] -= .5
                    Q[idx2][idx1] -= .5
                if val1 and not val2:
                    Q[idx1][idx1] += 1.
                    Q[idx1][idx2] -= .5
                    Q[idx2][idx1] -= .5
                if not val1 and not val2:
                    Q[idx1][idx2] += 1.
                    Q[idx2][idx1] += 1.
                    Q[idx1][idx1] -= 1.
                    Q[idx2][idx2] -= 1.

        return Q

    @classmethod
    def gen_problems(self, n_problems, size=(20, 25), **kwargs):
        # size: First is n_vars, second is number of clauses.
        # Want something like this:
        # [((1, True), (2, True), (3, True)), ((1, True), (2, False), (4, False))]
        n_vars = size[0] - 1
        problems = []
        for _ in range(n_problems):
            problem = []
            # Eg: 25 clauses
            for _ in range(size[1]):
                # Each has THREE tuples!! But random vars and random value.
                # (True/False).
                clause = (
                    (int(round(random.random() * n_vars)), bool(random.getrandbits(1))),
                    (int(round(random.random() * n_vars)), bool(random.getrandbits(1))),
                    (int(round(random.random() * n_vars)), bool(random.getrandbits(1)))
                )
                problem.append(sorted(clause))
            problems.append(sorted(problem))
        return [{"clauses": problem, "n_vars": size[0]} for problem in problems]
