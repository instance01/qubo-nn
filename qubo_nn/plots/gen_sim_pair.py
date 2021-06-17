import os
import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, max_len, req_len=None):
    mc = aggregate_single(paths, 'Total_Misclassifications', min_len, req_len=req_len, cutoff=max_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len, cutoff=max_len)
    return mc, eval_loss


def run():
    min_len = 1000
    req_len = 50
    base_paths = ['../runs/']
    problems = [
        ('sim_pair_60_40', 20000),
        ('sim_pair_50_50', 20000),
        ('sim_pair_50_40', 20000),
        ('sim_pair_40_40', 20000),
        ('sim_pair_40_30', 20000),
        ('sim_pair_30_30', 20000),
        ('sim_pair_20_20', 20000),
        ('sim_pair_10_10', 20000),
        ('sim_pair_5_5', 20000),
        ('sim_pair_0_0', 20000)
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
