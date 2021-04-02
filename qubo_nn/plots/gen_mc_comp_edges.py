import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len):
    arr_eval = aggregate_single(paths, 'Loss/Eval', min_len)
    arr_train = aggregate_single(paths, 'Loss/Train', min_len)
    arr_r2 = aggregate_single(paths, 'Custom/R2', min_len)
    return arr_eval, arr_train, arr_r2


def run():
    min_len = 1000
    base_path = '../runs/'
    problems = ['a19_2_r2', 'a19_2_r2_gen_edges2']
    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob('../runs/*-' + problem)
        data = aggregate(paths, min_len)
        kv[problem] = data

    with open('mc_comp_edges.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
