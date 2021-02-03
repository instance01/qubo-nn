import numpy as np
from networkx.generators.random_graphs import gnm_random_graph
from qubo_nn.problems.problem import Problem


class MaxCut(Problem):
    def __init__(self, graph):
        self.graph = graph

    def gen_qubo_matrix(self):

        n = self.graph.order()

        nodes = list(self.graph.nodes)

        Q = np.zeros((n, n))
        for edge in self.graph.edges:
            idx1 = nodes.index(edge[0])
            idx2 = nodes.index(edge[1])
            Q[idx1][idx1] += 1
            Q[idx2][idx2] += 1
            Q[idx1][idx2] -= 1
            Q[idx2][idx1] -= 1
        return Q

    @classmethod
    def gen_problems(self, n_problems, size=20, seed=None, **kwargs):
        if seed is not None:
            return [
                gnm_random_graph(*size, seed=seed)
                for _ in range(n_problems)
            ]
        return [gnm_random_graph(*size) for _ in range(n_problems)]
