import copy
import unittest
import networkx
from qubo_nn.problems import MinimumMaximumMatching


class TestMaxCut(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
        )
        want = [
            [-15, 9, 11, 11, 6],
            [9, -15, 11, 6, 11],
            [11, 11, -19, 11, 11],
            [11, 6, 11, -15, 9],
            [6, 11, 11, 9, -15]
        ]

        problem = MinimumMaximumMatching({}, copy.deepcopy(graph))
        matrix = problem.gen_qubo_matrix()
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        data = MinimumMaximumMatching.gen_problems(
            {}, 1, size=(20, 25), seed=1
        )
        self.assertCountEqual(data[0]["graph"].edges, [
            (0, 12), (0, 14), (0, 10), (0, 17), (0, 13), (0, 9), (2, 8),
            (3, 15), (3, 18), (3, 5), (3, 9), (4, 18), (6, 12), (6, 9), (7, 8),
            (7, 16), (7, 17), (7, 11), (7, 14), (9, 18), (10, 16), (13, 19),
            (13, 17), (13, 16), (14, 15)
        ])
