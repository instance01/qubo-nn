import itertools

import numpy as np

from qubo_nn.problems.problem import Problem
from qubo_nn.problems.util import gen_graph


class MinimumMaximumMatching(Problem):
    def __init__(self, cfg, graph, A=10, B=2, C=1):
        self.graph = graph
        self.A = A
        self.B = B
        self.C = C

    def gen_qubo_matrix(self):
        nodes = list(self.graph.nodes)
        edge_idx = {}
        for i, edge in enumerate(self.graph.edges):
            edge_idx[edge] = i
            edge_idx[edge[::-1]] = i

        n = len(edge_idx) // 2
        Q = np.zeros((n, n), dtype=np.dtype(np.int32))

        for vertex in nodes:
            for e1, e2 in itertools.combinations(self.graph.edges(vertex), 2):
                idx1 = edge_idx[e1]
                idx2 = edge_idx[e2]
                Q[idx1][idx2] += .5 * self.A
                Q[idx2][idx1] += .5 * self.A

        for edge in self.graph.edges:
            v1, v2 = edge
            for e1 in self.graph.edges(v1):
                idx1 = edge_idx[e1]
                Q[idx1][idx1] -= 2 * self.B

                for e2 in self.graph.edges(v2):
                    idx2 = edge_idx[e2]
                    Q[idx1][idx2] += .5 * self.B
                    Q[idx2][idx1] += .5 * self.B

            for e2 in self.graph.edges(v2):
                idx2 = edge_idx[e2]
                Q[idx2][idx2] -= 2 * self.B

                for e1 in self.graph.edges(v1):
                    idx1 = edge_idx[e1]
                    Q[idx1][idx2] += .5 * self.B
                    Q[idx2][idx1] += .5 * self.B

            idx = edge_idx[edge]
            Q[idx][idx] += self.C

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]


if __name__ == '__main__':
    G = MinimumMaximumMatching.gen_problems({}, 1, (4, 5))
    print(G[0]["graph"].edges)
    Q = MinimumMaximumMatching({}, **G[0]).gen_qubo_matrix()
    print(Q)

    for x in np.c_[tuple(i.ravel() for i in np.mgrid[:2, :2, :2, :2, :2])]:
        print(x, "|", x @ Q @ x.T)
