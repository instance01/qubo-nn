from qubo_nn.problems.number_partitioning import NumberPartitioning
from qubo_nn.problems.max_cut import MaxCut
from qubo_nn.problems.minimum_vertex_cover import MinimumVertexCover


PROBLEM_REGISTRY = {
    "NP": NumberPartitioning,
    "MC": MaxCut,
    "MVC": MinimumVertexCover
}
