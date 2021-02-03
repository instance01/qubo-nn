from qubo_nn.problems.number_partitioning import NumberPartitioning
from qubo_nn.problems.max_cut import MaxCut


PROBLEM_REGISTRY = {
    "NP": NumberPartitioning,
    "MC": MaxCut
}
