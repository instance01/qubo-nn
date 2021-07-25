import unittest
import networkx
from qubo_nn.problems import MaxIndependentSet


class TestMaxIndependentSet(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: My brain.
        """
        graph = networkx.Graph(
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        )
        problem = MaxIndependentSet({}, graph)
        matrix = problem.gen_qubo_matrix()
        want = [
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
        ]
        self.assertCountEqual(matrix.tolist(), want)
