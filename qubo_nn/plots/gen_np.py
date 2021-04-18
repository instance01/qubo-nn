import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, len_, req_len=None):
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len)
    return r2, eval_loss


def run():
    min_len = 200
    req_len = 20
    base_paths = ['../runs/', '../runs3/']
    problems = [
        ('np19_LONG_r2_5x5', 10),
        ('np19_LONG_r2_5x5_empty', 10),
        ('np19_LONG_r2', 10),
        ('np19_LONG_r2_64x64_2k', 10),
        ('np19_LONG_r2_64x64_5k', 10)
    ]
    kv = {}
    for problem in problems:
        print(problem)
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem[0]))
        print(len(paths))
        print(paths)
        data = aggregate(paths, min_len, problem[1], req_len=req_len)
        kv[problem[0]] = data

    with open('np_comp.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
