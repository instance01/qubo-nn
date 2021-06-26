import random
import unittest
from qubo_nn.problems import SetPartitioning


class TestSetPartitioning(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        problem = SetPartitioning(
            {},
            [1, 2, 3, 4],
            [[1, 4], [2, 4], [1, 2, 3], [3, 4], [2, 3], [1, 2, 4]],
            [3, 2, 1, 1, 3, 2]
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [-17, 10, 10, 10, 0, 20],
            [10, -18, 10, 10, 10, 20],
            [10, 10, -29, 10, 20, 20],
            [10, 10, 10, -19, 10, 10],
            [0, 10, 20, 10, -17, 10],
            [20, 20, 20, 10, 10, -28]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = SetPartitioning.gen_problems({}, 1, size=(20, 25))
        random.setstate(st0)
        self.assertCountEqual(
            data[0]["set_"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertCountEqual(
            data[0]["subsets"],
            [
                [0, 3, 4, 5, 8, 9, 11, 13, 14, 16, 19],
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18],
                [0, 1, 5, 8, 14, 15, 17, 18],
                [1, 2, 6, 7, 8, 9, 13, 14],
                [3, 4, 7, 15, 16, 19],
                [0, 1, 2, 6, 7, 8, 9, 15, 18, 19],
                [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18],
                [0, 3, 4, 5, 7, 11, 15, 17, 19],
                [0, 1, 4, 6, 9, 10, 11, 13, 17],
                [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 19],
                [3, 6, 8, 10, 11, 13, 17, 18, 19],
                [3, 5, 6, 7, 11, 14, 16, 17, 18],
                [1, 2, 5, 7, 8, 10, 11, 13, 14, 15, 16, 18],
                [2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 18, 19],
                [4, 6, 7, 8, 9, 11, 15, 16, 17, 18],
                [0, 1, 5, 6, 7, 8, 10, 11, 13, 15, 17],
                [1, 5, 6, 9, 11, 13, 14],
                [0, 1, 2, 5, 6, 12, 13, 14, 15, 16, 18],
                [0, 1, 2, 3, 5, 6, 8, 9, 13, 16, 17, 18, 19],
                [3, 5, 6, 9, 12, 15],
                [2, 3, 4, 7, 10, 12, 14, 18],
                [0, 5, 8, 10, 11, 12, 13, 16, 18, 19],
                [0, 2, 3, 4, 5, 9, 10, 12, 14],
                [3, 4, 5, 7, 13, 16, 18],
                [0, 1, 2, 3, 6, 8, 9, 10, 11, 17, 19]
            ]
        )
        self.assertCountEqual(
            data[0]["costs"],
            [0, 8, 7, 2, 5, 0, 5, 6, 7, 2, 8, 5, 3, 6, 8, 0, 6, 0, 4, 8, 9, 2, 4, 0, 7]
        )
