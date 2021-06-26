import numpy as np
from qubo_nn.problems.problem import Problem


class QuadraticKnapsack(Problem):
    def __init__(self, cfg, projects, budgets, constraint, P=10):
        self.projects = projects
        self.budgets = budgets.tolist()
        self.constraint = constraint
        self.P = P
        self.random_P = cfg["problems"]["QK"].get("random_P", False)

    def gen_qubo_matrix(self):
        # n_slack_vars = np.ceil(np.log2(self.constraint - min(self.budgets)))
        n_slack_vars = np.ceil(np.log2(self.constraint))
        self.budgets += [2 ** x for x in range(int(n_slack_vars))]
        n = len(self.budgets)

        if self.random_P:
            P = np.random.randint(5, 15)
        else:
            P = self.P

        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i][j] -= P * self.budgets[i] * self.budgets[j]
            Q[i][i] += P * 2 * self.constraint * self.budgets[i]

        for i in range(len(self.projects)):
            for j in range(len(self.projects)):
                Q[i][j] += self.projects[i][j]

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=4, constraint=30, **kwargs):
        high1 = cfg["problems"]["QK"].get("high1", 30)
        high2 = cfg["problems"]["QK"].get("high2", 30)

        print(cfg["problems"]["QK"])
        print("QK: Using following highs:", high1, high2)

        problems = []
        for _ in range(n_problems):
            projects = np.random.randint(low=1, high=high1, size=(size, size))
            projects = np.tril(projects) + np.tril(projects, -1).T

            problems.append((
                projects,
                np.random.randint(low=1, high=high2, size=(size,)),
                constraint
            ))
        return [
            {"projects": projects, "budgets": budgets, "constraint": constraint}
            for (projects, budgets, constraint) in problems
        ]
