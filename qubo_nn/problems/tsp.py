import numpy as np
from qubo_nn.problems.problem import Problem


class TSP(Problem):
    def __init__(self, dist_matrix, constraint, P=10):
        self.dist_matrix = dist_matrix
        self.constraint = constraint
        self.P = 10

    def gen_qubo_matrix(self):
        n = len(self.dist_matrix)
        Q = np.zeros((n ** 2, n ** 2))

        quadrants_y = list(range(0, n ** 2, n))
        quadrants_x = quadrants_y[1:] + [quadrants_y[0]]

        # The diagonal positive constraints
        for start_x in quadrants_y:
            for start_y in quadrants_y:
                for i in range(n):
                    Q[start_x + i][start_y + i] = 2 * self.constraint

        # The distance matrices
        for (start_x, start_y) in zip(quadrants_x, quadrants_y):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    Q[start_x + i][start_y + j] = self.P * self.dist_matrix[j][i]
                Q[start_x + i][start_y + i] = 2 * self.constraint

        # The middle diagonal negative constraints
        for start_x in quadrants_x:
            for i in range(n):
                Q[start_x + i][start_x + i] = -2 * self.constraint
                for j in range(n):
                    if i != j:
                        Q[start_x + i][start_x + j] += 2 * self.constraint

        return Q

    @classmethod
    def gen_problems(self, n_problems, size=4, **kwargs):
        # TODO: 200 and 100 (contraint) is hardcoded !!
        problems = []
        for _ in range(n_problems):
            dist_matrix = np.random.random(size=(size, size))
            np.fill_diagonal(dist_matrix, 0)
            constraint = np.random.random() * 200 + 100
            problems.append({"dist_matrix": dist_matrix.tolist(), "constraint": constraint})
        return problems
