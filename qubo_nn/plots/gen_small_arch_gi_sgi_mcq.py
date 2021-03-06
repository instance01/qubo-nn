import os
import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, max_len, req_len=None):
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len, cutoff=max_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len, cutoff=max_len)
    return r2, eval_loss


def run():
    min_len = 1000
    req_len = 30
    base_paths = ['../runs/']
    problems = [
        ('mcq_gen1', 20000),
        ('mcq_gen2', 20000),
        ('mcq_gen3', 20000),
        ('mcq_gen4', 20000),
        ('sgi_gen1_3', 20000),
        ('sgi_gen2_3', 20000),
        ('sgi_gen3_3', 20000),
        ('sgi_gen4_3', 20000),
        ('gi_gen1', 20000),
        ('gi_gen2', 20000),
        ('gi_gen3', 20000),
        ('gi_gen4', 20000)
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

    fname = os.path.splitext(__file__)[0][4:] + '.pickle'
    print("Saving in", fname)
    with open(fname, 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
