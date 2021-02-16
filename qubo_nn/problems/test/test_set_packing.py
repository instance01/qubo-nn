import random
import unittest
from qubo_nn.problems import SetPacking


class TestSetPacking(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        problem = SetPacking([1, 2], [[1, 2], [2], [1], [1]])
        matrix = problem.gen_qubo_matrix()
        want = [
            [1, -3, -3, -3],
            [-3, 1, 0, 0],
            [-3, 0, 1, -3],
            [-3, 0, -3, 1]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = SetPacking.gen_problems(1, size=(20, 25))
        random.setstate(st0)
        self.assertCountEqual(
            data[0]["set_"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertCountEqual(
            data[0]["subsets"],
            [
                [0, 3, 4, 5, 8, 9, 11, 13, 14, 16, 19],
                [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19],
                [2, 3, 7, 10, 16, 17, 19],
                [0, 4, 5, 9, 10, 11, 12, 16, 17],
                [3, 7, 8, 11, 19],
                [0, 3, 5, 6, 7, 11, 12, 13, 14],
                [0, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 17, 18, 19],
                [0, 1, 2, 3, 4, 7, 10, 11, 12, 14, 18],
                [2, 4, 6, 8, 9, 12, 14, 17, 18, 19],
                [1, 5, 10, 11, 12, 14, 15, 16, 18, 19],
                [0, 3, 4, 5, 8, 9, 13, 16, 18],
                [0, 1, 3, 7, 8, 9, 14, 16, 17, 18],
                [2, 5, 7, 8, 9, 13, 14, 17, 19],
                [0, 2, 3, 5, 6, 7, 8, 10, 12, 15, 16, 17, 18],
                [0, 1, 3, 5, 6, 7, 11, 12, 18],
                [0, 1, 2, 3, 5, 9, 10, 11, 12, 15, 16],
                [0, 1, 2, 3, 5, 6, 8, 10, 12, 15, 17],
                [1, 2, 5, 7, 9, 10, 17, 18, 19],
                [2, 3, 9, 10, 11, 12, 13, 15, 17, 18, 19],
                [0, 1, 3, 4, 6, 7, 11, 14, 15, 16, 17, 18],
                [2, 4, 5, 8, 11, 14],
                [2, 3, 4, 7, 10, 12, 14, 18],
                [1, 6, 9, 11, 12, 13, 14, 17, 19],
                [0, 1, 2, 4, 5, 6, 7, 11, 12, 14, 16],
                [2, 6, 7, 8, 10, 16, 19]
            ]
        )
