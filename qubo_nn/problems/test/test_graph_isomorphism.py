import unittest

import networkx
import numpy as np

from qubo_nn.problems import GraphIsomorphism


class TestGraphIsomorphism(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://researchspace.auckland.ac.nz/bitstream/handle/2292/31756/CDMTCS499.pdf?sequence=1
        """
        graph1 = networkx.Graph(
            [(0, 1), (1, 2)]
        )
        graph2 = networkx.Graph(
            [(0, 1), (0, 2)]
        )
        problem = GraphIsomorphism({}, graph1, graph2)
        matrix = problem.gen_qubo_matrix()
        want = [
            [
                [-2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [2.0, -2.0, 2.0, 0.0, 4.0, 2.0, 0.0, 2.0, 0.0],
                [2.0, 2.0, -2.0, 0.0, 2.0, 4.0, 0.0, 0.0, 2.0],
                [4.0, 0.0, 0.0, -2.0, 2.0, 2.0, 4.0, 0.0, 0.0],
                [0.0, 4.0, 2.0, 2.0, -2.0, 2.0, 0.0, 4.0, 2.0],
                [0.0, 2.0, 4.0, 2.0, 2.0, -2.0, 0.0, 2.0, 4.0],
                [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, -2.0, 2.0, 2.0],
                [0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 2.0, -2.0, 2.0],
                [0.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 2.0, -2.0]
            ]
        ]
        self.assertTrue(np.allclose(matrix, want))
