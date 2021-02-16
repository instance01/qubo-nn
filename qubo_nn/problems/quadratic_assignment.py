import numpy as np
from qubo_nn.problems.problem import Problem


class QuadraticAssignment(Problem):
    def __init__(self, flow_matrix, distance_matrix, P=200):
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.P = P

    def gen_qubo_matrix(self):
        n = len(self.flow_matrix)

        Q = np.zeros((n ** 2, n ** 2))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for m in range(n):
                        val = self.flow_matrix[i][j] * self.distance_matrix[k][m]
                        if val == 0:
                            Q[i * n + k][j * n + m] = self.P
                        else:
                            Q[i * n + k][j * n + m] = val

        for i in range(n ** 2):
            Q[i][i] = -self.P * 2

        return Q

    @classmethod
    def gen_problems(self, n_problems, size=3, **kwargs):
        # TODO: 50 is hardcoded !!!
        problems = []
        for _ in range(n_problems):
            problems.append((
                np.random.randint(low=0, high=50, size=(size, size)),
                np.random.randint(low=0, high=50, size=(size, size))
            ))
        return [
            {"flow_matrix": flow_matrix, "distance_matrix": distance_matrix}
            for (flow_matrix, distance_matrix) in problems
        ]
