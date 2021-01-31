import unittest
import networkx
from qubo_nn.problems import MaxCut


class TestMaxCut(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (3, 5)]
        )
        problem = MaxCut(graph)
        matrix = problem.gen_qubo_matrix()
        want = [
            [2, -1, -1, 0, 0],
            [-1, 2, 0, -1, 0],
            [-1, 0, 3, -1, -1],
            [0, -1, -1, 3, -1],
            [0, 0, -1, -1, 2]
        ]
        self.assertCountEqual(matrix.tolist(), want)
