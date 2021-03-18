import random
import unittest
from qubo_nn.problems import Max3SAT


class TestMax3SAT(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://canvas.auckland.ac.nz/courses/14782/files/574983/download?verifier=1xqRikUjTEBwm8PnObD8YVmKdeEhZ9Ui8axW8HwP&wrap=1
        """
        problem = Max3SAT(
            [
                ((0, True), (1, True), (2, True)),
                ((0, False), (1, True), (2, True)),
                ((0, True), (1, False), (2, True)),
                ((0, False), (1, True), (2, False))
            ],
            3
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [0.0, -1.0, 1.5, -0.5, 0.5, -0.5, 0.5],
            [-1.0, 1.0, 0.0, -0.5, -0.5, 0.5, -0.5],
            [1.5, 0.0, -1.0, -0.5, -0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5, 2.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 0.0],
            [-0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
            [0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = Max3SAT.gen_problems(1, size=(20, 25))
        random.setstate(st0)
        self.assertCountEqual(
            data[0]["clauses"],
            [
                [(0, False), (10, True), (12, False)],
                [(0, True), (10, True), (17, False)],
                [(1, True), (7, True), (14, False)],
                [(1, True), (11, False), (16, True)],
                [(2, False), (14, False), (17, True)],
                [(2, True), (10, True), (13, False)],
                [(3, True), (5, False), (15, False)],
                [(3, True), (7, False), (14, False)],
                [(3, True), (8, True), (17, True)],
                [(4, False), (4, False), (18, True)],
                [(4, False), (9, True), (11, True)],
                [(4, False), (12, True), (14, False)],
                [(4, False), (14, True), (17, True)],
                [(4, True), (16, True), (18, False)],
                [(5, True), (10, True), (16, False)],
                [(7, True), (16, True), (18, False)],
                [(8, False), (9, False), (11, False)],
                [(8, False), (9, False), (18, False)],
                [(9, False), (10, True), (15, False)],
                [(9, False), (13, True), (16, False)],
                [(9, False), (13, True), (18, False)],
                [(10, False), (10, True), (10, True)],
                [(11, False), (13, False), (14, False)],
                [(14, True), (16, True), (18, True)],
                [(15, False), (15, True), (18, True)]
            ]
        )
