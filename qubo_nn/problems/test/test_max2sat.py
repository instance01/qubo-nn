import random
import unittest
from qubo_nn.problems import Max2SAT


class TestMax2SAT(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        problem = Max2SAT(
            {"problems": {"M2SAT": {}}},
            [
                ((0, True), (1, True)),
                ((0, True), (1, False)),
                ((0, False), (1, True)),
                ((0, False), (1, False)),
                ((0, False), (2, True)),
                ((0, False), (2, False)),
                ((1, True), (2, False)),
                ((1, True), (3, True)),
                ((1, False), (2, True)),
                ((1, False), (2, False)),
                ((2, True), (3, True)),
                ((2, False), (3, False))
            ],
            4
        )
        matrix = problem.gen_qubo_matrix()
        want = [
            [1, 0, 0, 0],
            [0, 0, -.5, .5],
            [0, -.5, 0, 1],
            [0, .5, 1, -2]
        ]
        self.assertCountEqual(matrix.tolist(), want)

    def test_gen_problems(self):
        st0 = random.getstate()
        random.seed(1)
        data = Max2SAT.gen_problems(
            {"problems": {"M2SAT": {}}},
            1,
            size=(20, 25)
        )
        random.setstate(st0)
        self.assertCountEqual(
            data[0]["clauses"],
            [
                [(0, False), (14, False)],
                [(1, False), (16, True)],
                [(1, True), (13, True)],
                [(1, True), (17, False)],
                [(2, True), (6, True)],
                [(3, True), (10, True)],
                [(3, True), (16, False)],
                [(4, False), (4, True)],
                [(4, False), (8, True)],
                [(4, False), (8, True)],
                [(4, False), (14, True)],
                [(4, True), (19, False)],
                [(5, False), (9, False)],
                [(5, False), (15, True)],
                [(6, False), (9, False)],
                [(7, False), (9, False)],
                [(7, True), (13, False)],
                [(10, False), (11, False)],
                [(10, False), (18, True)],
                [(10, True), (15, True)],
                [(11, False), (16, True)],
                [(11, True), (17, True)],
                [(12, False), (15, False)],
                [(13, False), (16, False)],
                [(14, False), (18, True)]
            ]
        )
