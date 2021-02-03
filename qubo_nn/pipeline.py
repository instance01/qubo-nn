import numpy as np
from qubo_nn.problems import NumberPartitioning
from qubo_nn.problems import MaxCut
from qubo_nn.nn import Optimizer


class Classification:
    def __init__(self):
        pass

    def gen_qubo_matrices(self, cls, problem_size, n_problems=1000):
        problems = cls.gen_problems(problem_size, n_problems)
        qubo_matrices = [
            cls(problem).gen_qubo_matrix()
            for problem in problems
        ]
        return qubo_matrices

    def prep_data(self):
        problem_classes = [(NumberPartitioning, [20]), (MaxCut, [(20, 25)])]
        n_problems = 1000
        data = np.zeros(
            shape=(len(problem_classes) * n_problems, 20, 20),
            dtype=np.float32
        )
        labels = np.zeros(
            shape=(len(problem_classes) * n_problems,),
            dtype=np.long
        )
        for i, (cls, args) in enumerate(problem_classes):
            idx_start = i * n_problems
            idx_end = (i + 1) * n_problems
            qubo_matrices = self.gen_qubo_matrices(
                cls, *args, n_problems
            )
            qubo_matrices = np.array(qubo_matrices)
            data[idx_start:idx_end, :, :] = qubo_matrices
            labels[idx_start:idx_end] = i

        return data, labels

    def run_experiment(self):
        optimizer = Optimizer(*self.prep_data())
        optimizer.train()
        optimizer.eval()
