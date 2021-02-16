import numpy as np
from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class MinimumVertexCover(Problem):
    def __init__(self, graph, P=8):
        self.graph = graph
        self.P = P

    def gen_qubo_matrix(self):
        n = self.graph.order()
        nodes = list(self.graph.nodes)

        Q = np.zeros((n, n))
        for i in range(n):
            Q[i][i] += 1

        for edge in self.graph.edges:
            idx1 = nodes.index(edge[0])
            idx2 = nodes.index(edge[1])
            Q[idx1][idx1] -= self.P
            Q[idx2][idx2] -= self.P
            Q[idx1][idx2] += self.P / 2.
            Q[idx2][idx1] += self.P / 2.
        return Q

    @classmethod
    def gen_problems(self, n_problems, size=(20, 25), seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]
