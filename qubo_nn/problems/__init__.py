from qubo_nn.problems.number_partitioning import NumberPartitioning
from qubo_nn.problems.max_cut import MaxCut
from qubo_nn.problems.minimum_vertex_cover import MinimumVertexCover
from qubo_nn.problems.set_packing import SetPacking
from qubo_nn.problems.max2sat import Max2SAT
from qubo_nn.problems.set_partitioning import SetPartitioning
from qubo_nn.problems.graph_coloring import GraphColoring


PROBLEM_REGISTRY = {
    "NP": NumberPartitioning,
    "MC": MaxCut,
    "MVC": MinimumVertexCover,
    "SP": SetPacking,
    "M2SAT": Max2SAT,
    "SPP": SetPartitioning,
    "GC": GraphColoring
}
