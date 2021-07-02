import os
import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, max_len, req_len=None):
    r2 = aggregate_single(paths, 'Custom/A3_MC', min_len, req_len=req_len, cutoff=max_len)
    return r2, []


def run():
    min_len = 1000
    base_paths = ['../runs/']
    problems = ["TT1"]
    kv = {}
    for problem in problems:
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem))
        print(len(paths))
        print(paths)
        data = aggregate(paths, min_len, int(1e8), req_len=0)
        kv[problem[0]] = data

    fname = os.path.splitext(__file__)[0][4:] + '.pickle'
    print("Saving in", fname)
    with open(fname, 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
