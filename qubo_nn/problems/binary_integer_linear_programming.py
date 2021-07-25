import itertools
import numpy as np
from qubo_nn.problems.problem import Problem


class BinaryIntegerLinearProgramming(Problem):
    def __init__(self, cfg, c, S, b, A=10, B=2):
        self.c = c
        self.S = S
        self.b = b
        self.A = A
        self.B = B

    def gen_qubo_matrix(self):
        n = self.c.shape[0]
        m = self.b.shape[0]
        Q = np.zeros((n, n))

        # From Lucas 2014: The first term:
        for j in range(m):
            for i in range(n):
                Q[i][i] -= 2 * self.b[j] * self.S[j][i] * self.A
        for j in range(m):
            tuples = itertools.product(range(n), repeat=2)
            for tuple_ in tuples:
                x = tuple_[0]
                y = tuple_[1]
                Q[x][y] += self.S[j][x] * self.S[j][y] * self.A / 2.
                Q[y][x] += self.S[j][x] * self.S[j][y] * self.A / 2.

        # From Lucas 2014: The second term:
        for i in range(n):
            Q[i][i] -= self.c[i] * self.B

        return Q

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        # size=(m, n)
        m = size[0]
        n = size[1]
        high = cfg["problems"]["BIP"].get("high", 50)

        problems = []

        for _ in range(n_problems):
            x = np.random.randint(2, size=size[0])
            c = np.random.randint(high, size=(n,))
            S = np.random.randint(high, size=(m, n))
            b = S @ x
            problems.append({"c": c, "S": S, "b": b})

        return problems


if __name__ == "__main__":
    # x = np.array([1, 1, 0, 1, 1])
    c = np.array([4, 5, 1, 2, 3])
    S = np.array([
        [3, 4, 1, 1, 2],
        [1, 9, 2, 2, 1],
        [6, 8, 2, 2, 1]
    ])
    b = np.array([10, 13, 17])

    bip = BinaryIntegerLinearProgramming({}, c, S, b)
    Q = bip.gen_qubo_matrix()

    for x in np.c_[tuple(i.ravel() for i in np.mgrid[:2, :2, :2, :2, :2])]:
        print(x, "|", S @ x, "=?", b, "|", c @ x, "|", x @ Q @ x.T)
