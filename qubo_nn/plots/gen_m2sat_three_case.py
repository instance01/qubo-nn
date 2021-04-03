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
    min_len = 500
    base_path = '../runs/'
    problems = ['m2sat_16x16_5_F_v2', 'm2sat_16x16_5_F_v2_no_dupl_chk']
    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob('../runs/*-' + problem)

        paths = [path for path in paths if path not in [
            '../runs/21-04-03_13:01:10-8112457-datolith.cip.ifi.lmu.de-m2sat_16x16_5_F_v2',
            '../runs/21-04-02_07:40:21-3067619-feueropal.cip.ifi.lmu.de-m2sat_16x16_5_F_v2'
        ]]

        print(paths)
        if problem == 'm2sat_16x16_5_F_v2_no_dupl_chk':
            data = aggregate(paths, min_len, True)
        else:
            data = aggregate(paths, min_len, False)
        kv[problem] = data

    with open('m2sat_three_case.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
