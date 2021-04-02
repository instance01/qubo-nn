import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, len_):
    fpfn_tot_ratio = aggregate_single(paths, 'Custom/FPFN_TOT_Ratio', min_len, len_)
    r2 = aggregate_single(paths, 'Custom/R2', min_len, len_)
    return fpfn_tot_ratio, r2


def run():
    min_len = 201
    base_path = '../runs/'
    # problems = [
    #     ('m2sat_16x16_5_F', 201),
    #     ('m2sat_16x16_10_F', 61),
    #     ('m2sat_16x16_15_F', 51),
    #     ('m2sat_16x16_20_F', 51),
    #     ('m2sat_16x16_25_F', 31),
    #     ('m2sat_16x16_30_F', 31)
    # ]
    problems = [
        ('m2sat_16x16_5_F_v2', 201),
        ('m2sat_16x16_10_F_v2', 61),
        ('m2sat_16x16_15_F_v2', 51),
        ('m2sat_16x16_20_F_v2', 51),
        ('m2sat_16x16_25_F_v2', 31),
        ('m2sat_16x16_30_F_v2', 31)
    ]
    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob('../runs/*-' + problem[0])
        data = aggregate(paths, min_len, problem[1])
        kv[problem[0]] = data

    with open('m2sat_comp.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
