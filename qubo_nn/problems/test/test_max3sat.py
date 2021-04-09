import random
import unittest
from qubo_nn.problems import Max3SAT


class TestMax3SAT(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://canvas.auckland.ac.nz/courses/14782/files/574983/download?verifier=1xqRikUjTEBwm8PnObD8YVmKdeEhZ9Ui8axW8HwP&wrap=1
        """
        problem = Max3SAT(
            {"problems": {"M3SAT": {}}},
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
        data = Max3SAT.gen_problems(
            {"problems": {"M3SAT": {}}},
            1,
            size=(20, 25)
        )
        random.setstate(st0)
        self.assertCountEqual(
            data[0]["clauses"],
            [
                [(0, True), (1, False), (17, True)],
                [(0, True), (15, True), (18, True)],
                [(1, True), (8, False), (16, True)],
                [(1, True), (8, True), (17, False)],
                [(1, True), (10, False), (11, True)],
                [(1, True), (13, True), (19, False)],
                [(2, False), (10, True), (18, True)],
                [(3, False), (15, False), (16, False)],
                [(3, True), (4, False), (12, False)],
                [(3, True), (7, False), (16, False)],
                [(4, False), (5, True), (18, True)],
                [(4, False), (8, False), (9, False)],
                [(4, True), (10, True), (16, False)],
                [(5, True), (13, True), (15, False)],
                [(6, True), (14, True), (14, True)],
                [(7, False), (8, True), (13, True)],
                [(7, False), (9, False), (14, False)],
                [(7, True), (8, True), (13, True)],
                [(7, True), (16, True), (17, True)],
                [(8, False), (15, False), (18, False)],
                [(8, True), (14, True), (18, True)],
                [(10, False), (12, False), (12, True)],
                [(11, True), (12, True), (16, False)],
                [(13, False), (17, True), (18, True)],
                [(13, True), (14, False), (18, False)]
            ]
        )
