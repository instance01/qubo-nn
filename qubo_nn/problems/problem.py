# TODO: I probably won't use abc here due to a possibility of using Cython.

class Problem:
    def gen_qubo_matrix(self):
        pass

    # TODO Make sure overriding this works.
    @classmethod
    def gen_problems(self, cfg, n_problems, size=20, **kwargs):
        pass
