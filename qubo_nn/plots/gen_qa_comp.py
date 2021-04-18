import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, len_, req_len=None):
    fpfn_tot_ratio = aggregate_single(paths, 'Custom/FPFN_TOT_Ratio', min_len, req_len=req_len)
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len)
    return fpfn_tot_ratio, r2, eval_loss


def run():
    min_len = 200
    req_len = 22
    base_paths = ['../runs/', '../runs5/']
    problems = [
        ('qa_N_9_norm2_1M', 10),
        ('qa_N_16_norm1', 10),
        ('qa_N_16_norm2', 10),
        ('qa_N_16_norm3', 10),
        ('qa_N_64_norm1', 10),
        ('qa_N_64_norm2', 10),
        ('qa_N_64_norm3', 10),
        ('qa_N_100_norm1', 10),
        ('qa_N_100_norm2', 10),
        ('qa_N_100_norm3', 10),
        ('qa_N_144_norm3', 10),

        ('qa_N_16_norm2_1M', 10),
        ('qa_N_16_norm2_4M', 10),

        ('qa_special_loss1', 10),

        ('qa_N_9_norm2_1M_2', 10)
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

    with open('qa_comp.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
