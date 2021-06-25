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
    base_paths = ['../runs/']
    problems = [
        ('sp_gen1_big_arch', 100, 100),
        ('sp_gen2_big_arch', 100, 100),
        ('sp_gen3_big_arch', 100, 100),
        ('sp_gen4_big_arch', 100, 100),
        ('m2sat_8x8_10_gen1_big_arch', 90, 20),
        ('m2sat_8x8_10_gen2_big_arch', 90, 20),
        ('m2sat_8x8_10_gen3_big_arch', 90, 20),
        ('m2sat_8x8_10_gen4_big_arch', 90, 20),
        ('qa_N_144_norm3_generalized_50k_gen1_big_arch', 20000, 10),
        ('qa_N_144_norm3_generalized_50k_gen2_big_arch', 20000, 10),
        ('qa_N_144_norm3_generalized_50k_gen3_big_arch', 20000, 10),
        ('qa_N_144_norm3_generalized_50k_gen4_big_arch', 20000, 10),
        ('np19_LONG_generalized_gen1_big_arch', 5000, 3000),
        ('np19_LONG_generalized_gen2_big_arch', 5000, 2000),
        ('np19_LONG_generalized_gen3_big_arch', 5000, 3000),
        ('np19_LONG_generalized_gen4_big_arch', 5000, 3000)
    ]
    kv = {}
    for problem in problems:
        print(problem)
        paths = []
        for base_path in base_paths:
            paths.extend(glob.glob(base_path + '*-' + problem[0]))
        print(len(paths))
        print(paths)
        data = aggregate(paths, min_len, problem[1], req_len=problem[2])
        kv[problem[0]] = data

    fname = os.path.splitext(__file__)[0][4:] + '.pickle'
    print("Saving in", fname)
    with open(fname, 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
