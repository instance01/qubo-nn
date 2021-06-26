from networkx.generators.random_graphs import gnm_random_graph


def gen_graph(n_problems, size, seed=None):
    if seed is not None:
        return [
            gnm_random_graph(*size, seed=seed)
            for _ in range(n_problems)
        ]
    return [gnm_random_graph(*size) for _ in range(n_problems)]
