import unittest
from qubo_nn.problems import NumberPartitioning


class TestNumberPartitioning(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        problem = NumberPartitioning([25, 7, 13, 31, 42, 17, 21, 10])
        matrix = problem.gen_qubo_matrix()
        want = [
            [-3525, 175, 325, 775, 1050, 425, 525, 250],
            [175, -1113, 91, 217, 294, 119, 147, 70],
            [325, 91, -1989, 403, 546, 221, 273, 130],
            [775, 217, 403, -4185, 1302, 527, 651, 310],
            [1050, 294, 546, 1302, -5208, 714, 882, 420],
            [425, 119, 221, 527, 714, -2533, 357, 170],
            [525, 147, 273, 651, 882, 357, -3045, 210],
            [250, 70, 130, 310, 420, 170, 210, -1560],
        ]
        self.assertCountEqual(matrix.tolist(), want)
