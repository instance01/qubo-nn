import itertools
import numpy as np
from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class GraphIsomorphism(Problem):
    def __init__(self, cfg, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2

    def gen_qubo_matrix(self):
        n = self.graph1.order()

        Q = np.zeros((n ** 2, n ** 2))

        indices = []

        for i in range(n):
            for j in range(n):
                idx = i * n + j
                Q[idx][idx] -= 1
            for k, m in itertools.combinations(list(range(n)), 2):
                idx1 = i * n + k
                idx2 = i * n + m
                Q[idx1][idx2] += 2
                Q[idx2][idx1] += 2

        for j in range(n):
            for i in range(n):
                idx = i * n + j
                Q[idx][idx] -= 1
            for k, m in itertools.combinations(list(range(n)), 2):
                idx1 = k * n + j
                idx2 = m * n + j
                Q[idx1][idx2] += 2
                Q[idx2][idx1] += 2

        nodes = list(self.graph1.nodes)
        for edge in self.graph1.edges:
            i = nodes.index(edge[0])
            j = nodes.index(edge[1])
            # eg 0,1

            for i_dash in range(n):
                idx1 = i * n + i_dash
                for j_dash in range(n):
                    idx2 = j * n + j_dash
                    if (i_dash, j_dash) in self.graph2.edges:
                        continue
                    Q[idx1][idx2] += 1
                    Q[idx2][idx1] += 1
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs1 = gen_graph(n_problems, size, seed)
        graphs2 = gen_graph(n_problems, size, seed)
        return [
            {"graph1": graph1, "graph2": graph2}
            for graph1, graph2 in zip(graphs1, graphs2)
        ]
