import numpy as np
from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class MaxCut(Problem):
    def __init__(self, cfg, graph):
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
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]
