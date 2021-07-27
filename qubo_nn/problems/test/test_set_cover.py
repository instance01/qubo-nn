import random
import unittest
from qubo_nn.problems import SetCover


class TestSetCover(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        set_ = [0, 1, 2]
        subsets = [[0, 1], [2]]
        want = [
            [10.0, 0.0, -4.0, -8.0, -4.0, -8.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, -4.0, -8.0],
            [-4.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            [-8.0, 0.0, 12.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0, 0.0, 12.0, 12.0, 0.0, 0.0],
            [0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
            [0.0, -8.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0]
        ]
        problem = SetCover(
            {"problems": {"SC": {}}},
            SetCover.gen_matrix(set_, subsets)
        )
        matrix = problem.gen_qubo_matrix()
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = SetCover.gen_problems(
            {"problems": {"SC": {}}},
            1,
            size=(10, 10)
        )
        random.setstate(st0)
        subset_matrix_want = [
            [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
            [1., 0., 1., 1., 0., 0., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 0., 0., 0.],
            [1., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0., 1., 0., 0., 1., 1.],
            [0., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
            [1., 0., 1., 1., 1., 1., 1., 1., 0., 0.]
        ]

        self.assertCountEqual(
            data[0]["subset_matrix"].tolist(),
            subset_matrix_want
        )
