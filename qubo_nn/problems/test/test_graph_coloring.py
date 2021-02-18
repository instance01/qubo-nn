import unittest
import networkx
from qubo_nn.problems import GraphColoring


class TestGraphColoring(unittest.TestCase):
    def test_gen_qubo_matrix(self):
        """Test whether a correct QUBO is generated.

        Test case from: https://arxiv.org/pdf/1811.11538.pdf
        """
        graph = networkx.Graph(
            [(1, 2), (2, 3), (3, 4), (2, 5), (1, 5), (4, 5), (4, 2)]
        )
        problem = GraphColoring(graph, 3)
        matrix = problem.gen_qubo_matrix()
        want = [
            [-4., 4., 4., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
            [ 4.,-4., 4., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.],
            [ 4., 4.,-4., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
            [ 2., 0., 0.,-4., 4., 4., 2., 0., 0., 2., 0., 0., 2., 0., 0.],
            [ 0., 2., 0., 4.,-4., 4., 0., 2., 0., 0., 2., 0., 0., 2., 0.],
            [ 0., 0., 2., 4., 4.,-4., 0., 0., 2., 0., 0., 2., 0., 0., 2.],
            [ 0., 0., 0., 2., 0., 0.,-4., 4., 4., 2., 0., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 2., 0., 4.,-4., 4., 0., 2., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 2., 4., 4.,-4., 0., 0., 2., 0., 0., 0.],
            [ 0., 0., 0., 2., 0., 0., 2., 0., 0.,-4., 4., 4., 2., 0., 0.],
            [ 0., 0., 0., 0., 2., 0., 0., 2., 0., 4.,-4., 4., 0., 2., 0.],
            [ 0., 0., 0., 0., 0., 2., 0., 0., 2., 4., 4.,-4., 0., 0., 2.],
            [ 2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 0., 0.,-4., 4., 4.],
            [ 0., 2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 0., 4.,-4., 4.],
            [ 0., 0., 2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 4., 4.,-4.]
        ]
        self.assertCountEqual(matrix.tolist(), want)