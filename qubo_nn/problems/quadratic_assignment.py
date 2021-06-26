import numpy as np
from qubo_nn.problems.problem import Problem


class QuadraticAssignment(Problem):
    def __init__(self, cfg, flow_matrix, distance_matrix, P=200):
        self.flow_matrix = flow_matrix
        self.distance_matrix = distance_matrix
        self.P = P

        if cfg["problems"]["QA"].get("debug", False):
            self.P = 0.

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
    def gen_problems(self, cfg, n_problems, size=3, **kwargs):
        high = cfg["problems"]["QA"].get("high", 50)  # Outdated.
        debug = cfg["problems"]["QA"].get("debug", False)

        problems = []
        for _ in range(n_problems):
            if debug:
                x = np.arange(1, 50)
            else:
                x = np.arange(1, 1000)

            choice = np.random.choice(x, size=(2, size, size), replace=False)
            flow = choice[0]
            dist = choice[1]

            if debug:
                dist[1][0] = 1.
                flow[1][0] = 1.

            # flow = np.random.randint(low=1, high=high, size=(size, size))
            # dist = np.random.randint(low=1, high=high, size=(size, size))

            np.fill_diagonal(flow, 0)
            np.fill_diagonal(dist, 0)
            problems.append((
                np.tril(flow) + np.tril(flow, -1).T,
                np.tril(dist) + np.tril(dist, -1).T,
            ))
        return [
            {"flow_matrix": flow_matrix, "distance_matrix": distance_matrix}
            for (flow_matrix, distance_matrix) in problems
        ]
