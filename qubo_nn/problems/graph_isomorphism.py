import itertools
import numpy as np
from qubo_nn.problems.subgraph_isomorphism import SubGraphIsomorphism
from qubo_nn.problems.util import gen_graph


class GraphIsomorphism(SubGraphIsomorphism):
    def __init__(self, cfg, graph1, graph2):
        super(GraphIsomorphism, self).__init__(cfg, graph1, graph2, a=1, b=2)

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs1 = gen_graph(n_problems, size, seed)
        graphs2 = gen_graph(n_problems, size, seed)
        return [
            {"graph1": graph1, "graph2": graph2}
            for graph1, graph2 in zip(graphs1, graphs2)
        ]
