import unittest

import numpy as np

from qubo_nn.problems import BinaryIntegerLinearProgramming


class TestBinaryIntegerLinearProgramming(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        c = np.array([4, 5, 1, 2, 3])
        S = np.array([
            [3, 4, 1, 1, 2],
            [1, 9, 2, 2, 1],
            [6, 8, 2, 2, 1]
        ])
        b = np.array([10, 13, 17])
        problem = BinaryIntegerLinearProgramming(
            {"problems": {"BIP": {}}},
            c, S, b
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [-2448.0, 690.0, 170.0, 170.0, 130.0],
            [690.0, -4260.0, 380.0, 380.0, 250.0],
            [170.0, 380.0, -1312.0, 90.0, 60.0],
            [170.0, 380.0, 90.0, -1314.0, 60.0],
            [130.0, 250.0, 60.0, 60.0, -946.0]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = np.random.get_state()
        np.random.seed(1)
        data = BinaryIntegerLinearProgramming.gen_problems(
            {"problems": {"BIP": {}}},
            1,
            size=(10, 10)
        )
        np.random.set_state(st0)
        s_matrix_want = [
            [20, 11, 42, 28, 29, 14,  4, 23, 23, 41],
            [49, 30, 32, 22, 13, 41,  9,  7, 22,  1],
            [0,  17,  8, 24, 13, 47, 42,  8, 30,  7],
            [3,   6, 21, 49,  3,  4, 24, 49, 43, 12],
            [26, 16, 45, 41, 18, 15,  0,  4, 25, 47],
            [34, 23,  7, 26, 25, 40, 22,  9,  3, 39],
            [23, 36, 27, 37, 19, 38,  8, 32, 34, 10],
            [23, 15, 47, 23, 25,  7, 28, 10, 46, 32],
            [24, 23,  1, 49, 13,  3,  0, 13,  6, 21],
            [6,   2, 12, 27, 21, 11,  7, 13,  8, 11]
        ]
        c_vector_want = [16,  1, 12,  7, 45,  6, 25, 20, 37, 18]
        b_vector_want = [124, 171, 157, 132, 104, 156, 190, 154,  82,  68]

        self.assertTrue(np.allclose(data[0]["S"], s_matrix_want))
        self.assertCountEqual(data[0]["c"].tolist(), c_vector_want)
        self.assertCountEqual(data[0]["b"].tolist(), b_vector_want)
