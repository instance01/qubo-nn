import unittest
import numpy as np
from qubo_nn.problems import QuadraticKnapsack


class TestQuadraticKnapsack(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        projects = np.array([
            [2, 4, 3, 5],
            [4, 5, 1, 3],
            [3, 1, 2, 2],
            [5, 3, 2, 4]
        ])
        budgets = np.array([8, 6, 5, 3])
        problem = QuadraticKnapsack(projects, budgets, 16, 10)
        matrix = problem.gen_qubo_matrix()
        want = [
            [1922.,-476.,-397.,-235., -80.,-160.,-320.,-640.],
            [-476.,1565.,-299.,-177., -60.,-120.,-240.,-480.],
            [-397.,-299.,1352.,-148., -50.,-100.,-200.,-400.],
            [-235.,-177.,-148., 874., -30., -60.,-120.,-240.],
            [ -80., -60., -50., -30., 310., -20., -40., -80.],
            [-160.,-120.,-100., -60., -20., 600., -80.,-160.],
            [-320.,-240.,-200.,-120., -40., -80.,1120.,-320.],
            [-640.,-480.,-400.,-240., -80.,-160.,-320.,1920.]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = np.random.get_state()
        np.random.seed(1)
        data = QuadraticKnapsack.gen_problems(1, size=4)
        np.random.set_state(st0)
        self.assertCountEqual(
            data[0].tolist(),
            [
                [37, 43, 12,  8],
                [ 9, 11,  5, 15],
                [ 0, 16,  1, 12],
                [ 7, 45,  6, 25]
            ]
        )
        self.assertCountEqual(
            data[1].tolist(),
            [20, 37, 18, 20]
        )
