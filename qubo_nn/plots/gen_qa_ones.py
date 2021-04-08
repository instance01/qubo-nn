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
    problems = ['qa_N_16_norm3', 'qa_N_16_norm3_goddamn']
    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob(base_path + '*-' + problem)

        print(paths)
        data = aggregate(paths, min_len, False)
        kv[problem] = data

    with open('qa_ones.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
