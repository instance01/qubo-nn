import numpy as np
from qubo_nn.problems.problem import Problem


class QuadraticKnapsack(Problem):
    def __init__(self, projects, budgets, constraint, P):
        self.projects = projects
        self.budgets = budgets.tolist()
        self.constraint = constraint
        self.P = P

    def gen_qubo_matrix(self):
        n_slack_vars = np.ceil(np.log2(self.constraint - min(self.budgets)))
        self.budgets += [2 ** x for x in range(int(n_slack_vars))]
        n = len(self.budgets)

        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i][j] -= self.P * self.budgets[i] * self.budgets[j]
            Q[i][i] += self.P * 2 * self.constraint * self.budgets[i]

        for i in range(len(self.projects)):
            for j in range(len(self.projects)):
                Q[i][j] += self.projects[i][j]

        return Q

    @classmethod
    def gen_problems(self, n_problems, size=4, constraint=16, **kwargs):
        # TODO: 50 is hardcoded !!!
        return (
            np.random.randint(low=0, high=50, size=(size, size)),
            np.random.randint(low=0, high=50, size=(size,)),
            np.random.randint(low=0, high=50)
        )
