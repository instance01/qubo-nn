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


cfg = {
    "problems": {
        "NP": {
            "size": 16
        },
        "MC": {
            "size": [16, 20]
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



prob = NumberPartitioning.gen_problems(cfg, 1, **cfg["problems"]["NP"])
qubo = NumberPartitioning(cfg, **prob[0]).gen_qubo_matrix()
print("NP")
print(qubo)

prob = MaxCut.gen_problems(cfg, 1, **cfg["problems"]["MC"])
qubo = MaxCut(cfg, **prob[0]).gen_qubo_matrix()
print("MC")
print(qubo)

prob = MinimumVertexCover.gen_problems(cfg, 1, **cfg["problems"]["MVC"])
qubo = MinimumVertexCover(cfg, **prob[0]).gen_qubo_matrix()
print("MVC")
print(qubo)

prob = SetPacking.gen_problems(cfg, 1, **cfg["problems"]["SP"])
qubo = SetPacking(cfg, **prob[0]).gen_qubo_matrix()
print("SP")
print(qubo)

prob = Max2SAT.gen_problems(cfg, 1, **cfg["problems"]["M2SAT"])
qubo = Max2SAT(cfg, **prob[0]).gen_qubo_matrix()
print("M2SAT")
print(qubo)

prob = SetPartitioning.gen_problems(cfg, 1, **cfg["problems"]["SPP"])
qubo = SetPartitioning(cfg, **prob[0]).gen_qubo_matrix()
print("SPP")
print(qubo)

prob = GraphColoring.gen_problems(cfg, 1, **cfg["problems"]["GC"])
qubo = GraphColoring(cfg, **prob[0]).gen_qubo_matrix()
print("GC")
print(qubo)

prob = QuadraticAssignment.gen_problems(cfg, 1, **cfg["problems"]["QA"])
qubo = QuadraticAssignment(cfg, **prob[0]).gen_qubo_matrix()
print("QA")
print(qubo)

prob = QuadraticKnapsack.gen_problems(cfg, 1, **cfg["problems"]["QK"])
qubo = QuadraticKnapsack(cfg, **prob[0]).gen_qubo_matrix()
print("QK")
print(qubo)

prob = Max3SAT.gen_problems(cfg, 1, **cfg["problems"]["M3SAT"])
qubo = Max3SAT(cfg, **prob[0]).gen_qubo_matrix()
print("M3SAT")
print(qubo)

prob = TSP.gen_problems(cfg, 1, **cfg["problems"]["TSP"])
qubo = TSP(cfg, **prob[0]).gen_qubo_matrix()
print("TSP")
print(qubo)
