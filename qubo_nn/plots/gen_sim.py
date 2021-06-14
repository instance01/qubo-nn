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
    req_len = 30
    base_paths = ['../runs/']
    problems = [
        ('sim_6_4', 20000),
        ('sim_6_3', 20000),
        ('sim_5_4', 20000),
        ('sim_4_4', 20000),
        ('sim_4_3', 20000),
        ('sim_3_3', 20000),
        ('sim_2_2', 20000),
        ('sim_1_1', 20000),
        ('sim_0_0', 20000)
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
