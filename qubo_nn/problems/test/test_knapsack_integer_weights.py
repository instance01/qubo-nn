import unittest
import numpy as np
from qubo_nn.problems import KnapsackIntegerWeights


class TestKnapsackIntegerWeights(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: My brain.
        """
        w = np.array([2, 5, 3])
        c = np.array([5, 2, 4])
        W = 7

        problem = KnapsackIntegerWeights(
            {"problems": {"KIW": {}}},
            w, c, W
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [35.0, 100.0, 60.0, -20.0, -40.0, -60.0, -80.0, -100.0, -120.0, -140.0],
            [100.0, 248.0, 150.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0, -350.0],
            [55.0, 148.0, 82.0, -30.0, -60.0, -90.0, -120.0, -150.0, -180.0, -210.0],
            [-20.0, -50.0, -30.0, 0.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [-40.0, -100.0, -60.0, 30.0, 30.0, 70.0, 90.0, 110.0, 130.0, 150.0],
            [-60.0, -150.0, -90.0, 40.0, 70.0, 80.0, 130.0, 160.0, 190.0, 220.0],
            [-80.0, -200.0, -120.0, 50.0, 90.0, 130.0, 150.0, 210.0, 250.0, 290.0],
            [-100.0, -250.0, -150.0, 60.0, 110.0, 160.0, 210.0, 240.0, 310.0, 360.0],
            [-120.0, -300.0, -180.0, 70.0, 130.0, 190.0, 250.0, 310.0, 350.0, 430.0],
            [-140.0, -350.0, -210.0, 80.0, 150.0, 220.0, 290.0, 360.0, 430.0, 480.0]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = np.random.get_state()
        np.random.seed(1)
        data = KnapsackIntegerWeights.gen_problems(
            {"problems": {"KIW": {}}},
            1,
            size=(5, 5)
        )
        np.random.set_state(st0)
        w_want = [37, 43, 12,  8,  9]
        c_want = [11,  5, 15,  0, 16]

        self.assertCountEqual(data[0]["w"].tolist(), w_want)
        self.assertCountEqual(data[0]["c"].tolist(), c_want)
