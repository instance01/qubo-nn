import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, len_, req_len=None):
    r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len)
    eval_loss = aggregate_single(paths, 'Loss/Eval', min_len, req_len=req_len)
    return r2, eval_loss


def run():
    min_len = 200
    req_len = 200
    base_paths = ['../runs/']
    problems = [
        ('qa_N_64_norm3_5p', 10),
        ('qa_N_64_norm3_10p', 10),
        ('qa_N_64_norm3_15p', 10),
        ('qa_N_64_norm3_20p', 10),
        ('qa_N_64_norm3_25p', 10),
        ('qa_N_64_norm3_30p', 10),
        ('qa_N_64_norm3_35p', 10),
        ('qa_N_64_norm3_40p', 10),
        ('qa_N_64_norm3_45p', 10),
        ('qa_N_64_norm3_50p', 10),
        ('qa_N_64_norm3_55p', 10),
        ('qa_N_64_norm3_60p', 10),
        ('qa_N_64_norm3_65p', 10),
        ('qa_N_64_norm3_70p', 10),
        ('qa_N_64_norm3_75p', 10),
        ('qa_N_64_norm3_80p', 10),
        ('qa_N_64_norm3_85p', 10),
        ('qa_N_64_norm3_90p', 10)
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

    with open('qk_zeros.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
