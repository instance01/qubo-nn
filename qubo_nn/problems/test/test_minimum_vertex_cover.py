import unittest
import networkx
from qubo_nn.problems import MinimumVertexCover


class TestMinimumVertexCover(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (3, 5)]
        )
        problem = MinimumVertexCover({}, graph)
        matrix = problem.gen_qubo_matrix()
        want = [
            [-15, 4, 4, 0, 0],
            [4, -15, 0, 4, 0],
            [4, 0, -23, 4, 4],
            [0, 4, 4, -23, 4],
            [0, 0, 4, 4, -15]
        ]
        self.assertCountEqual(matrix.tolist(), want)
