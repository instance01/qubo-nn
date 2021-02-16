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

    def test_gen_problems(self):
        data = MaxCut.gen_problems(1, size=(20, 25), seed=1)
        self.assertCountEqual(data[0]["graph"].edges, [
            (0, 12), (0, 14), (0, 10), (0, 17), (0, 13), (0, 9), (2, 8),
            (3, 15), (3, 18), (3, 5), (3, 9), (4, 18), (6, 12), (6, 9), (7, 8),
            (7, 16), (7, 17), (7, 11), (7, 14), (9, 18), (10, 16), (13, 19),
            (13, 17), (13, 16), (14, 15)
        ])
