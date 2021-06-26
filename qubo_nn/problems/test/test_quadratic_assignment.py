import unittest
import numpy as np
from qubo_nn.problems import QuadraticAssignment


class TestQuadraticAssignment(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        X = np.array([
            [0, 5, 2],
            [5, 0, 3],
            [2, 3, 0]
        ])
        Y = np.array([
            [0, 8, 15],
            [8, 0, 13],
            [15, 13, 0]
        ])
        problem = QuadraticAssignment({"problems": {"QA": {}}}, X, Y, 200)
        matrix = problem.gen_qubo_matrix()
        want = [
            [-400., 200., 200., 200.,  40.,  75., 200.,  16.,  30.],
            [ 200.,-400., 200.,  40., 200.,  65.,  16., 200.,  26.],
            [ 200., 200.,-400.,  75.,  65., 200.,  30.,  26., 200.],
            [ 200.,  40.,  75.,-400., 200., 200., 200.,  24.,  45.],
            [  40., 200.,  65., 200.,-400., 200.,  24., 200.,  39.],
            [  75.,  65., 200., 200., 200.,-400.,  45.,  39., 200.],
            [ 200.,  16.,  30., 200.,  24.,  45.,-400., 200., 200.],
            [  16., 200.,  26.,  24., 200.,  39., 200.,-400., 200.],
            [  30.,  26., 200.,  45.,  39., 200., 200., 200.,-400.]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = np.random.get_state()
        np.random.seed(1)
        data = QuadraticAssignment.gen_problems(
            {"problems": {"QA": {"debug": True}}},
            1,
            size=4
        )
        np.random.set_state(st0)
        self.assertCountEqual(
            data[0]["flow_matrix"].tolist(),
            [
                [0, 1, 46, 22],
                [1, 0, 31, 36],
                [46, 31, 0, 20],
                [22, 36, 20, 0]
            ]
        )
        self.assertCountEqual(
            data[0]["distance_matrix"].tolist(),
            [
                [0, 1, 24, 11],
                [1, 0, 5, 29],
                [24, 5, 0, 48],
                [11, 29, 48, 0]
            ]
        )
