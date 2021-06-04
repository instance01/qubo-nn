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
    base_paths = ['../runs/', '../runs3/', '../runs5/', '../runs7/']
    problems = [
        ('np19_LONG_generalized_gen2_small_arch', 20000),
        ('tsp_gen2', 150),
        ('qa_N_144_norm3_generalized_50k_gen2_small_arch', 20000),
        ('a19_2_generalized_gen2_small_arch', 20000),
        ('mvc3_generalized_gen2_small_arch', 20000),
        ('gc1_generalized_gen2_small_arch', 20000),

        ('np19_LONG_generalized_gen3_small_arch', 20000),
        ('tsp_gen3', 150),
        ('qa_N_144_norm3_generalized_50k_gen3_small_arch', 20000),
        ('a19_2_generalized_gen3_small_arch', 20000),
        ('mvc3_generalized_gen3_small_arch', 20000),
        ('gc1_generalized_gen3_small_arch', 20000),

        ('np19_LONG_generalized_gen4_small_arch', 20000),
        ('tsp_gen4', 150),
        ('qa_N_144_norm3_generalized_50k_gen4_small_arch', 20000),
        ('a19_2_generalized_gen4_small_arch', 20000),
        ('mvc3_generalized_gen4_small_arch', 20000),
        ('gc1_generalized_gen4_small_arch', 20000),

        ('a19_2_r2', 20000),
        ('mvc3_r2', 20000),
        ('gc1_r2', 20000),
        ('np19_LONG_r2', 20000),
        ('tsp_gen1', 150),
        ('qa_N_144_norm3', 20000),

        ('a19_2_generalized_gen2_small_arch_2', 20000),
        ('mvc3_generalized_gen2_small_arch_2', 20000),
        ('gc1_generalized_gen2_small_arch_2', 20000),
        ('a19_2_generalized_gen3_small_arch_2', 20000),
        ('mvc3_generalized_gen3_small_arch_2', 20000),
        ('gc1_generalized_gen3_small_arch_2', 20000),
        ('a19_2_generalized_gen4_small_arch_2', 20000),
        ('mvc3_generalized_gen4_small_arch_2', 20000),
        ('gc1_generalized_gen4_small_arch_2', 20000)
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
