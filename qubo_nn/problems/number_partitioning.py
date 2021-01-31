import numpy as np
from qubo_nn.problems.problem import Problem


class NumberPartitioning(Problem):
    def __init__(self, numbers):
        self.numbers = numbers

    def gen_qubo_matrix(self):
        n = len(self.numbers)
        c = sum(self.numbers)

        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i][j] = self.numbers[i] * (self.numbers[i] - c)
                else:
                    Q[i][j] = self.numbers[i] * self.numbers[j]
        return Q
