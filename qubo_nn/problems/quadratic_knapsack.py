import numpy as np
from qubo_nn.problems.problem import Problem


class QuadraticKnapsack(Problem):
    def __init__(self, projects, budgets, constraint, P=10):
        self.projects = projects
        self.budgets = budgets.tolist()
        self.constraint = constraint
        self.P = P

    def gen_qubo_matrix(self):
        # n_slack_vars = np.ceil(np.log2(self.constraint - min(self.budgets)))
        n_slack_vars = np.ceil(np.log2(self.constraint))
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
    def gen_problems(self, n_problems, size=4, constraint=30, **kwargs):
        # TODO: 30, 50 is hardcoded !!!
        problems = []
        for _ in range(n_problems):
            problems.append((
                np.random.randint(low=0, high=30, size=(size, size)),
                np.random.randint(low=0, high=30, size=(size,)),
                # np.random.randint(low=30, high=50)
                constraint
            ))
        return [
            {"projects": projects, "budgets": budgets, "constraint": constraint}
            for (projects, budgets, constraint) in problems
        ]
