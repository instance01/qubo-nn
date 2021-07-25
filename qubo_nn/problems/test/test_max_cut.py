import copy
import unittest
import networkx
from qubo_nn.problems import MaxCut
from qubo_nn.problems.max_cut import MaxCutMemoryEfficient


class TestMaxCut(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4)]
        )
        want = [
            [-2, 1, 1, 0, 0],
            [1, -2, 0, 1, 0],
            [1, 0, -3, 1, 1],
            [0, 1, 1, -3, 1],
            [0, 0, 1, 1, -2]
        ]

        problem = MaxCut({}, copy.deepcopy(graph))
        matrix = problem.gen_qubo_matrix()
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        data = MaxCut.gen_problems({}, 1, size=(20, 25), seed=1)
        self.assertCountEqual(data[0]["graph"].edges, [
            (0, 12), (0, 14), (0, 10), (0, 17), (0, 13), (0, 9), (2, 8),
            (3, 15), (3, 18), (3, 5), (3, 9), (4, 18), (6, 12), (6, 9), (7, 8),
            (7, 16), (7, 17), (7, 11), (7, 14), (9, 18), (10, 16), (13, 19),
            (13, 17), (13, 16), (14, 15)
        ])


class TestMaxCutMemoryEfficient(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4)]
        )
        want = [
            [-2, 1, 1, 0, 0],
            [1, -2, 0, 1, 0],
            [1, 0, -3, 1, 1],
            [0, 1, 1, -3, 1],
            [0, 0, 1, 1, -2]
        ]
        problem = MaxCutMemoryEfficient({}, graph.edges, 5)
        matrix = problem.gen_qubo_matrix()
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        data = MaxCutMemoryEfficient.gen_problems({}, 1, size=(20, 25), seed=1)
        self.assertCountEqual(data[0]["edge_list"], [
            (0, 12), (0, 14), (0, 10), (0, 17), (0, 13), (0, 9), (2, 8),
            (3, 15), (3, 18), (3, 5), (3, 9), (4, 18), (6, 12), (6, 9), (7, 8),
            (7, 16), (7, 17), (7, 11), (7, 14), (9, 18), (10, 16), (13, 19),
            (13, 17), (13, 16), (14, 15)
        ])
