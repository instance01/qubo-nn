from qubo_nn.problems.number_partitioning import NumberPartitioning
from qubo_nn.problems.max_cut import MaxCut
from qubo_nn.problems.minimum_vertex_cover import MinimumVertexCover
from qubo_nn.problems.set_packing import SetPacking
from qubo_nn.problems.max2sat import Max2SAT
from qubo_nn.problems.set_partitioning import SetPartitioning
from qubo_nn.problems.graph_coloring import GraphColoring
from qubo_nn.problems.quadratic_assignment import QuadraticAssignment
from qubo_nn.problems.quadratic_knapsack import QuadraticKnapsack
from qubo_nn.problems.max3sat import Max3SAT
from qubo_nn.problems.tsp import TSP
from qubo_nn.problems.graph_isomorphism import GraphIsomorphism
from qubo_nn.problems.subgraph_isomorphism import SubGraphIsomorphism
from qubo_nn.problems.max_clique import MaxClique
from qubo_nn.problems.exact_cover import ExactCover
from qubo_nn.problems.binary_integer_linear_programming import BinaryIntegerLinearProgramming  # noqa


PROBLEM_REGISTRY = {
    "NP": NumberPartitioning,
    "MC": MaxCut,
    "MVC": MinimumVertexCover,
    "SP": SetPacking,
    "M2SAT": Max2SAT,
    "SPP": SetPartitioning,
    "GC": GraphColoring,
    "QA": QuadraticAssignment,
    "QK": QuadraticKnapsack,
    "M3SAT": Max3SAT,
    "TSP": TSP,
    "GI": GraphIsomorphism,
    "SGI": SubGraphIsomorphism,
    "MCQ": MaxClique,
    "EC": ExactCover,
    "BIP": BinaryIntegerLinearProgramming
}
