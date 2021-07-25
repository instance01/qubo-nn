import random
import unittest
from qubo_nn.problems import ExactCover


class TestExactCover(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: My brain.
        """
        set_ = [0, 1, 2, 3, 4]
        subsets = [[0, 1], [2], [3, 4], [4]]
        problem = ExactCover(
            {"problems": {"EC": {}}},
            ExactCover.gen_matrix(set_, subsets)
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [-4., 0., 0., 0.],
            [0., -2., 0., 0.],
            [0., 0., -2., 1.],
            [0., 0., 1., 0.]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = ExactCover.gen_problems(
            {"problems": {"EC": {}}},
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
