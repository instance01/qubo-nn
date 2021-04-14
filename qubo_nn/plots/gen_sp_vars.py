import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, cutoff=False):
    arr_eval = aggregate_single(paths, 'Loss/Eval', min_len)
    arr_train = aggregate_single(paths, 'Loss/Train', min_len)
    if cutoff:
        arr_r2 = aggregate_single(paths, 'Custom/R2', min_len, cutoff=6)
    else:
        arr_r2 = aggregate_single(paths, 'Custom/R2', min_len)
    return arr_eval, arr_train, arr_r2


def run():
    min_len = 100
    base_path = '../runs/'
    problems = [
        'sp4', 'sp5', 'sp6', 'sp4_sort', 'sp5_sort', 'sp6_sort',
        'sp4_100k', 'sp5_100k', 'sp6_100k', 'sp4_sort_100k', 'sp5_sort_100k', 'sp6_sort_100k',
        'sp4_1M'
    ]
    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob(base_path + '*-' + problem)

        print(paths)
        data = aggregate(paths, min_len, False)
        kv[problem] = data

    with open('sp_vars.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
