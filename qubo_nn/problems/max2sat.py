import random
import collections
import numpy as np
from qubo_nn.problems.problem import Problem


class Max2SAT(Problem):
    def __init__(self, cfg, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars

    def gen_qubo_matrix(self):
        n = self.n_vars
        Q = np.zeros((n, n))

        # Clauses is a list of 2-tuples of 2-tuples.
        # Eg: [((1, True), (2, True)), ((1, True), (2, False))]
        # So we loop through all tuples and find the type of it.
        # There are 4 types: (True, True), (True, False), etc.
        for clause in self.clauses:
            idx1 = clause[0][0]
            idx2 = clause[1][0]
            val1 = clause[0][1]
            val2 = clause[1][1]

            if val1 and val2:
                Q[idx1][idx1] -= 1
                Q[idx2][idx2] -= 1
                Q[idx1][idx2] += .5
                Q[idx2][idx1] += .5
            elif val1 and not val2:
                Q[idx2][idx2] += 1
                Q[idx1][idx2] -= .5
                Q[idx2][idx1] -= .5
            elif not val1 and val2:
                Q[idx1][idx1] += 1
                Q[idx1][idx2] -= .5
                Q[idx2][idx1] -= .5
            elif not val1 and not val2:
                Q[idx1][idx2] += .5
                Q[idx2][idx1] += .5
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(20, 25), **kwargs):
        check_duplicates = cfg["problems"]["M2SAT"].get("check_duplicates", False)
        if check_duplicates:
            print("M2SAT gen_problems: Checking for duplicates.")

        # size: First is n_vars, second is number of clauses.
        # Want something like this:
        # [((1, True), (2, True)), ((1, True), (2, False))]
        n_vars = size[0] - 1
        problems = []
        for _ in range(n_problems):
            problem = []

            assoc = collections.defaultdict(list)

            # Eg: 25 clauses
            for _ in range(size[1]):
                # Each has TWO tuples!! But random vars and random value
                # (True/False).
                idx1 = int(round(random.random() * n_vars))
                idx2 = int(round(random.random() * n_vars))
                val1 = bool(random.getrandbits(1))
                val2 = bool(random.getrandbits(1))

                # TODO: With the checks below we sometimes end up with less than the desired # of clauses.

                if check_duplicates:
                    # TODO Added Mar 19 06:45
                    if idx1 == idx2:
                        continue

                    if not val1:
                        idx1 = -idx1
                    if not val2:
                        idx2 = -idx2

                    cont = False
                    for x in assoc[-idx1]:
                        for y in assoc[-idx2]:
                            if -x == y:
                                cont = True

                    if cont:
                        continue

                    assoc[idx1].append(idx2)
                    assoc[idx2].append(idx1)

                clause = (
                    (abs(idx1), val1),
                    (abs(idx2), val2)
                )
                clause = sorted(clause)

                if check_duplicates:
                    if clause in problem:
                        continue

                    # TODO Added Mar 19 07:45
                    other_clause1 = [clause[0], (clause[1][0], not clause[1][1])]
                    other_clause2 = [(clause[0][0], not clause[0][1]), clause[1]]
                    other_clause3 = [(clause[0][0], not clause[0][1]), (clause[1][0], not clause[1][1])]
                    if other_clause1 in problem or other_clause2 in problem or other_clause3 in problem:
                        continue

                problem.append(clause)
            problems.append(sorted(problem))
        return [{"clauses": problem, "n_vars": size[0]} for problem in problems]
