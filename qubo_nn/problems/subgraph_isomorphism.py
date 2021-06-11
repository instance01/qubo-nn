import itertools
import numpy as np
from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class SubGraphIsomorphism(Problem):
    """Generalization of Graph Isomorphism."""
    def __init__(self, cfg, graph1, graph2, a=1, b=2):
        self.graph1 = graph1
        self.graph2 = graph2
        self.a = a
        self.b = b

    def gen_qubo_matrix(self):
        n1 = self.graph1.order()
        n2 = self.graph2.order()

        Q = np.zeros((n1 * n2, n1 * n2))

        for i in range(n1):
            for j in range(n2):
                idx = i * n2 + j
                Q[idx][idx] -= 1 * self.a
            for k, m in itertools.combinations(list(range(n2)), 2):
                idx1 = i * n2 + k
                idx2 = i * n2 + m
                Q[idx1][idx2] += 2 * self.a
                Q[idx2][idx1] += 2 * self.a

        for j in range(n2):
            for i in range(n1):
                idx = i * n2 + j
                Q[idx][idx] -= 1
            for k, m in itertools.combinations(list(range(n1)), 2):
                idx1 = k * n2 + j
                idx2 = m * n2 + j
                Q[idx1][idx2] += 2
                Q[idx2][idx1] += 2

        nodes = list(self.graph1.nodes)
        for edge in self.graph1.edges:
            i = nodes.index(edge[0])
            j = nodes.index(edge[1])

            for i_dash in range(n2):
                idx1 = i * n2 + i_dash
                for j_dash in range(n2):
                    idx2 = j * n2 + j_dash
                    if (i_dash, j_dash) in self.graph2.edges:
                        continue
                    Q[idx1][idx2] += 1 * self.b
                    Q[idx2][idx1] += 1 * self.b
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size1, size2, seed=None, **kwargs):
        # NOTE: size1 should be smaller than size2 or equal to it.
        graphs1 = gen_graph(n_problems, size1, seed)
        graphs2 = gen_graph(n_problems, size2, seed)
        return [
            {"graph1": graph1, "graph2": graph2}
            for graph1, graph2 in zip(graphs1, graphs2)
        ]
