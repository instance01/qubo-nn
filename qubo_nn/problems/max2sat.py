import random
import numpy as np
from qubo_nn.problems.problem import Problem


class Max2SAT(Problem):
    def __init__(self, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars

    def gen_qubo_matrix(self):
        n = self.n_vars
        Q = np.zeros((n, n))

        # Clauses is a list of 2-tuples of 2-tuples.
        # Eg: [((1, True), (2, True)), ((1, True), (2, False))]
        # So we loop through all tuples and find the type of it.
        # There are 4 types: (True, True), (True, False), etc.
        for clause in self.clauses:
            idx1 = clause[0][0]
            idx2 = clause[1][0]
            val1 = clause[0][1]
            val2 = clause[1][1]

            if val1 and val2:
                Q[idx1][idx1] -= 1
                Q[idx2][idx2] -= 1
                Q[idx1][idx2] += .5
                Q[idx2][idx1] += .5
            elif val1 and not val2:
                Q[idx2][idx2] += 1
                Q[idx1][idx2] -= .5
                Q[idx2][idx1] -= .5
            elif not val1 and val2:
                Q[idx1][idx1] += 1
                Q[idx1][idx2] -= .5
                Q[idx2][idx1] -= .5
            elif not val1 and not val2:
                Q[idx1][idx2] += .5
                Q[idx2][idx1] += .5
        return Q

    @classmethod
    def gen_problems(self, n_problems, size=(20, 25), **kwargs):
        # size: First is n_vars, second is number of clauses.
        # Want something like this:
        # [((1, True), (2, True)), ((1, True), (2, False))]
        n_vars = size[0] - 1
        problems = []
        for _ in range(n_problems):
            problem = []
            # Eg: 25 clauses
            for _ in range(size[1]):
                # Each has TWO tuples!! But random vars and random value
                # (True/False).
                clause = (
                    (int(round(random.random() * n_vars)), bool(random.getrandbits(1))),
                    (int(round(random.random() * n_vars)), bool(random.getrandbits(1)))
                )
                problem.append(clause)
            problems.append(problem)
        return [{"clauses": problem, "n_vars": size[0]} for problem in problems]
