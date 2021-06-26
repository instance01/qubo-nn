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
import numpy as np
import qubovert


def solve_qubo2(item):
    qubo_size = 8  # TODO Hardcoded!

    Q = qubovert.utils.matrix_to_qubo(item.reshape(qubo_size, qubo_size))
    sol = Q.solve_bruteforce(all_solutions=False)
    sol_ = [0 for _ in range(qubo_size)]
    for k, v in sol.items():
        sol_[k] = v
    return sol_


cfg = {
    "problems": {
        "NP": {
            "size": 16
        },
        "MC": {
            "size": [8, 10]
        },
        "MVC": {
            "size": [16, 20]
        },
        "SP": {
            "size": [20, 16]
        },
        "M2SAT": {
            "size": [16, 20]
        },
        "SPP": {
            "size": [20, 16]
        },
        "GC": {
            "size": [4, 6],
            "n_colors": 4
        },
        "QA": {
            "size": 4
        },
        "QK": {
            "size": 10,
            "constraint": 40
        },
        "M3SAT": {
            "size": [6, 10]
        },
        "TSP": {
            "size": 4
        }
    }
}



prob = MaxCut.gen_problems(cfg, 1, **cfg["problems"]["MC"])
qubo = MaxCut(cfg, **prob[0]).gen_qubo_matrix()
qubo = -qubo
print("MC")
print(qubo)
print(solve_qubo2(qubo))
x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
print(x.T @ qubo @ x)
import pdb; pdb.set_trace()
