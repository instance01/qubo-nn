import random
import unittest
from qubo_nn.problems import Max2SAT


class TestMax2SAT(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        problem = Max2SAT(
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
        data = Max2SAT.gen_problems(1, size=(20, 25))
        random.setstate(st0)
        self.assertCountEqual(data,
            [[
                ((3, True), (20, False)),
                ((6, False), (19, False)),
                ((16, True), (5, False)),
                ((1, True), (10, True)),
                ((19, False), (17, False)),
                ((18, False), (15, False)),
                ((23, False), (1, True)),
                ((14, True), (22, True)),
                ((5, False), (18, True)),
                ((6, False), (23, True)),
                ((6, False), (17, True)),
                ((11, False), (23, False)),
                ((21, True), (23, False)),
                ((5, True), (18, False)),
                ((3, False), (22, True)),
                ((18, True), (24, True)),
                ((21, True), (5, False)),
                ((15, True), (12, True)),
                ((13, True), (21, False)),
                ((6, True), (10, True)),
                ((4, True), (22, True)),
                ((17, False), (2, True)),
                ((13, True), (4, True)),
                ((10, False), (18, False)),
                ((1, True), (21, True))
            ]]
        )
