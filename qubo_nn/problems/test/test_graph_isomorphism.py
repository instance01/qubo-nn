import unittest
import networkx
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
            [-2.,   2.,   2.,   3.,   0.,   0.,   2.,   0.,   0.],
            [ 2.,  -2.,   2.,   0.,   3.,   1.,   0.,   2.,   0.],
            [ 2.,   2.,  -2.,   0.,   1.,   3.,   0.,   0.,   2.],
            [ 3.,   0.,   0.,  -2.,   2.,   2.,   3.,   0.,   0.],
            [ 0.,   3.,   1.,   2.,  -2.,   2.,   0.,   3.,   1.],
            [ 0.,   1.,   3.,   2.,   2.,  -2.,   0.,   1.,   3.],
            [ 2.,   0.,   0.,   3.,   0.,   0.,  -2.,   2.,   2.],
            [ 0.,   2.,   0.,   0.,   3.,   1.,   2.,  -2.,   2.],
            [ 0.,   0.,   2.,   0.,   1.,   3.,   2.,   2.,  -2.]
        ]
        self.assertCountEqual(matrix.tolist(), want)
