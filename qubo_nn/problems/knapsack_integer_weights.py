import numpy as np
from qubo_nn.problems.problem import Problem


class KnapsackIntegerWeights(Problem):
    def __init__(self, cfg, w, c, W, A=10, B=1):
        self.w = w
        self.c = c
        self.W = W
        self.A = A
        self.B = B

    def gen_qubo_matrix(self):
        N = self.w.shape[0]
        Q = np.zeros((N + self.W, N + self.W))

        # First term.
        for i in range(self.W):
            Q[N + i][N + i] -= 2 * self.A
            for j in range(self.W):
                Q[N + i][N + j] += .5 * self.A
                Q[N + j][N + i] += .5 * self.A

        # Second term.
        for i in range(self.W):
            for j in range(self.W):
                Q[N + i][N + j] += .5 * (i+1) * (j+1) * self.A
                Q[N + j][N + i] += .5 * (i+1) * (j+1) * self.A

        for i in range(self.W):
            for j in range(N):
                Q[N + i][j] -= (i+1) * self.w[j] * self.A
                Q[j][N + i] -= (i+1) * self.w[j] * self.A

        for i in range(N):
            for j in range(N):
                Q[i][j] += .5 * self.w[i] * self.w[j] * self.A
                Q[j][i] += .5 * self.w[i] * self.w[j] * self.A

        # Last term.
        for i in range(N):
            Q[i][i] -= self.c[i] * self.B
            Q[j][i] -= self.c[i] * self.B

        return Q

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        high = cfg["problems"]["KIW"].get("high", 50)

        problems = []

        for _ in range(n_problems):
            w = np.random.randint(0, high, size=(size[0],))
            c = np.random.randint(0, high, size=(size[0],))
            problems.append({"w": w, "c": c, "W": size[1]})

        return problems


if __name__ == "__main__":
    w = np.array([2, 5, 3])
    c = np.array([5, 2, 4])
    W = 7
    sc = KnapsackIntegerWeights(
        {"problems": {"KIW": {}}},
        w, c, W
    )
    Q = sc.gen_qubo_matrix()

    print(Q.tolist())

    for x in np.c_[tuple(i.ravel() for i in np.mgrid[:2, :2, :2, :2, :2, :2, :2, :2, :2, :2])]:  # noqa
        if x @ Q @ x.T < 0:
            print(x, "|", x @ Q @ x.T)
