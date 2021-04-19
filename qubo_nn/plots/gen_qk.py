import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, len_, req_len=None):
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len)
    return r2, eval_loss


def run():
    min_len = 200
    req_len = 10
    base_paths = ['../runs/']
    problems = [
        ('qk7_6x6_high30_1M', 10),
        ('qk7_6x6_high10_1M', 10),
        ('qk7_6x6_high30_1M_multiply', 10),
        ('qk8_16x16_high30_1M', 10),
        ('qk8_16x16_high30_1M_multiply', 10),
        ('qk8_6x6_high10_1M', 10)
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

    problems = [
        ('qk10', 10),
        ('qk10_norm', 10),
        ('qk10_high10', 10),
        ('qk10_high10_1', 10)
    ]
    req_len = 200
    for problem in problems:
        print(problem)
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem[0]))
        print(len(paths))
        print(paths)
        data = aggregate(paths, min_len, problem[1], req_len=req_len)
        kv[problem[0]] = data

    with open('qk.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
