import os
import glob
import pickle

from lib import aggregate_single


problems_short = ["np", "mc", "mvc", "sp", "m2sat", "spp", "gc", "qa", "qk", "m3sat", "tsp", "gi", "sgi", "mcq"]


def aggregate(paths, min_len, max_len, req_len=None):
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len, cutoff=max_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len, cutoff=max_len)
    return r2, eval_loss


def run():
    min_len = 1000
    base_paths = ['../runs/']
    problems = []
    for short in problems_short:
        for i in range(1, 20):
            problems.append(
                ("red_" + short + "_" + str(i), 20000)
            )
    kv = {}
    for problem in problems:
        print(problem)
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem[0]))
        print(len(paths))
        print(paths)
        req_len = 150
        if "gc" in problem[0] or "qa" in problem[0] or "mcq" in problem[0]:
            req_len = 10
        data = aggregate(paths, min_len, problem[1], req_len=req_len)
        kv[problem[0]] = data

    fname = os.path.splitext(__file__)[0][4:] + '.pickle'
    print("Saving in", fname)
    with open(fname, 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
