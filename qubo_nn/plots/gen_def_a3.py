import os
import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, max_len, req_len=None):
    # r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len, cutoff=max_len)
    # eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len, cutoff=max_len)
    train_loss = aggregate_single(paths, 'Loss/Train', min_len, req_len=req_len, cutoff=max_len)
    return [], train_loss


def run():
    min_len = 1000
    base_paths = ['../runs/']
    problems = [
        ('a3_1', 5),
        ('a3_2', 5000),
        ('a3_3', 5000)
    ]
    kv = {}
    for problem in problems:
        print(problem)
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem[0]))
        print(len(paths))
        print(paths)
        data = aggregate(paths, min_len, 20000, req_len=problem[1])
        kv[problem[0]] = data

    fname = os.path.splitext(__file__)[0][4:] + '.pickle'
    print("Saving in", fname)
    with open(fname, 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
