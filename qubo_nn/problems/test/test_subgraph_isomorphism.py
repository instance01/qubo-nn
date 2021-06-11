import unittest

import networkx

from qubo_nn.problems import SubGraphIsomorphism


class TestSubGraphIsomorphism(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: My brain. Tested whether solution is sensible with
        qbsolv.
        """
        graph1 = networkx.Graph(
            [(0, 1), (1, 2)]
        )
        graph2 = networkx.Graph(
            [(0, 1), (0, 2), (2, 3)]
        )
        problem = SubGraphIsomorphism({}, graph1, graph2)
        matrix = problem.gen_qubo_matrix()
        want = [
            [-2.,   2.,   2.,   2.,   4.,   0.,   0.,   2.,   2.,   0.,   0.,   0.],
            [ 2.,  -2.,   2.,   2.,   0.,   4.,   2.,   2.,   0.,   2.,   0.,   0.],
            [ 2.,   2.,  -2.,   2.,   0.,   2.,   4.,   0.,   0.,   0.,   2.,   0.],
            [ 2.,   2.,   2.,  -2.,   2.,   2.,   0.,   4.,   0.,   0.,   0.,   2.],
            [ 4.,   0.,   0.,   2.,  -2.,   2.,   2.,   2.,   4.,   0.,   0.,   2.],
            [ 0.,   4.,   2.,   2.,   2.,  -2.,   2.,   2.,   0.,   4.,   2.,   2.],
            [ 0.,   2.,   4.,   0.,   2.,   2.,  -2.,   2.,   0.,   2.,   4.,   0.],
            [ 2.,   2.,   0.,   4.,   2.,   2.,   2.,  -2.,   2.,   2.,   0.,   4.],
            [ 2.,   0.,   0.,   0.,   4.,   0.,   0.,   2.,  -2.,   2.,   2.,   2.],
            [ 0.,   2.,   0.,   0.,   0.,   4.,   2.,   2.,   2.,  -2.,   2.,   2.],
            [ 0.,   0.,   2.,   0.,   0.,   2.,   4.,   0.,   2.,   2.,  -2.,   2.],
            [ 0.,   0.,   0.,   2.,   2.,   2.,   0.,   4.,   2.,   2.,   2.,  -2.]
        ]
        self.assertCountEqual(matrix.tolist(), want)
