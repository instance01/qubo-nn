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
    req_len = 80
    base_paths = ['../runs/']
    problems = [
        ('v_mvc_gen1', 20000),
        ('v_mvc_gen2', 20000),
        ('v_mvc_gen3', 20000),
        ('v_mvc_gen4', 20000),
        ('v_gc_gen1', 20000),
        ('v_gc_gen2', 20000),
        ('v_gc_gen3', 20000),
        ('v_gc_gen4', 20000),
        ('v_np_gen1', 20000),
        ('v_np_gen2', 20000),
        ('v_np_gen3', 20000),
        ('v_np_gen4', 20000),
        ('v_sp_gen1', 20000),
        ('v_sp_gen2', 20000),
        ('v_sp_gen3', 20000),
        ('v_sp_gen4', 20000)
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
