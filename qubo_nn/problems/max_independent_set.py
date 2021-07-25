import numpy as np
from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class MaxIndependentSet(Problem):
    def __init__(self, cfg, graph):
        self.graph = graph

    def gen_qubo_matrix(self):
        n = self.graph.order()

        Q = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i][i] = -1
                # NOTE: Only difference to MaxClique is in the lines below:
                if i < j and (i, j) in self.graph.edges:
                    Q[i][j] = 1
                    Q[j][i] = 1

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]
