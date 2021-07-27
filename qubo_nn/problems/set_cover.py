import random
import numpy as np
from qubo_nn.problems.problem import Problem


class SetCover(Problem):
    def __init__(self, cfg, subset_matrix, A=4, B=2):
        self.subset_matrix = subset_matrix
        self.A = A
        self.B = B

    def gen_qubo_matrix(self):
        n = self.subset_matrix.shape[0]
        N = self.subset_matrix.shape[1]
        Q = np.zeros((n * N + N, n * N + N))

        for i in range(n):
            for m in range(N):
                idx = N + i * N + m
                Q[idx][idx] -= self.A * 2
                for m2 in range(N):
                    idx2 = N + i * N + m2
                    Q[idx][idx2] += .5 * self.A
                    Q[idx2][idx] += .5 * self.A

        for i in range(n):
            for m in range(N):
                idx = N + i * N + m
                # Q[idx][idx] += self.A * m * m * .5
                for m2 in range(N):
                    idx2 = N + i * N + m2
                    Q[idx][idx2] += self.A * ((m+1) * (m2+1)) / 2.
                    Q[idx2][idx] += self.A * ((m+1) * (m2+1)) / 2.

                # NOTE: We now use m2 as index, this is intentional.
                for m2 in range(N):
                    if self.subset_matrix[i][m2] > 0:
                        Q[idx][m2] -= self.A * (m+1)  # * N
                        Q[m2][idx] -= self.A * (m+1)  # * N

            for m2 in range(N):
                if self.subset_matrix[i][m2] > 0:
                    for m in range(N):
                        if self.subset_matrix[i][m] > 0:
                            Q[m][m2] += self.A * .5  # * N
                            Q[m2][m] += self.A * .5  # * N

        for i in range(N):
            Q[i][i] += self.B

        return Q

    # TODO: This is 100% copy&paste from SetPacking... Refactor!!
    @classmethod
    def gen_matrix(cls, set_, subsets):
        B = np.zeros((len(set_), len(subsets)))

        for m, x in enumerate(set_):
            for i, subset in enumerate(subsets):
                if x in subset:
                    B[m][i] = 1

        return B

    # TODO: This is 100% copy&paste from SetPacking... Refactor!!
    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        sorting = cfg["problems"]["SC"].get("sorting", False)

        problems = []

        uniques = set()

        set_ = list(range(size[0]))
        for _ in range(n_problems * 3):
            subsets = set()
            for _ in range(size[1]):
                x = list(filter(lambda x: random.random() < 0.5, set_))
                if not x:
                    continue
                subsets.add(tuple(x))
            if len(subsets) != size[1]:
                continue
            subsets = sorted(list(subsets))
            if tuple(subsets) in uniques:
                continue
            uniques.add(tuple(subsets))

            B = SetCover.gen_matrix(set_, subsets)

            # Sort it.
            if sorting:
                y = np.array([2 ** i for i in range(len(subsets))])
                z = B @ y
                idx = np.argsort(z)
                B = B[idx]

            problems.append(B)
            if len(problems) == n_problems:
                break

        print("SC generated problems:", len(problems))
        return [{"subset_matrix": matrix} for matrix in problems]


if __name__ == "__main__":
    # set_ = [0, 1, 2, 3, 4]
    # subsets = [[0, 1], [2], [3, 4], [4]]
    set_ = [0, 1, 2]
    subsets = [[0, 1], [2]]
    sc = SetCover(
        {"problems": {"SC": {}}},
        SetCover.gen_matrix(set_, subsets)
    )
    Q = sc.gen_qubo_matrix()

    print(set_)
    print(subsets)
    print(SetCover.gen_matrix(set_, subsets))
    print(Q.tolist())

    for x in np.c_[tuple(i.ravel() for i in np.mgrid[:2, :2, :2, :2, :2, :2, :2, :2])]:  # noqa
        if x @ Q @ x.T < 0:
            print(x, "|", x @ Q @ x.T)
