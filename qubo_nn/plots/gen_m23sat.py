import glob
import pickle

from lib import aggregate_single


def aggregate(paths, min_len, req_len):
    arr_eval = aggregate_single(paths, 'Loss/Eval', min_len)
    arr_train = aggregate_single(paths, 'Loss/Train', min_len)
    arr_r2 = aggregate_single(paths, 'Custom/R2', min_len, req_len=req_len)
    return arr_eval, arr_train, arr_r2


def run():
    min_len = 100
    base_path = '../runs/'
    problems = [
        'm3sat_5_F',
        'm3sat_10_F',
        'm3sat_15V_5_F',
        'm3sat_15V_10_F',
        'm3sat_20V_10_F',
        'm3sat_5V_5_F',
        'm3sat_5V_3_F',
        'm3sat_5V_2_F',

        'm3sat_10V_5_A',
        'm3sat_10V_10_A',
        'm3sat_15V_5_A',
        'm3sat_15V_10_A',
        'm3sat_5V_5_A',
        'm3sat_5V_3_A',
        'm3sat_5V_2_A',
        'm3sat_5V_5_A_2',

        'm2sat_16x16_10_F_v2_1M',
        'm2sat_16x16_5_F_v2_1M',
        'm2sat_16x16_15_F_v2_1M',
        'm2sat_16x16_20_F_v2_1M',
        'm2sat_16x16_25_F_v2_1M',
        'm2sat_16x16_30_F_v2_1M',

        'm2sat_8x8_10_F_v2_1M',
        'm2sat_8x8_5_F_v2_1M',
        'm2sat_8x8_15_F_v2_1M',
        'm2sat_8x8_20_F_v2_1M',
        'm2sat_8x8_25_F_v2_1M',
        'm2sat_8x8_30_F_v2_1M'
    ]

    req_len = 10

    kv = {}
    for problem in problems:
        print(problem)
        paths = glob.glob(base_path + '*-' + problem)

        print(paths)
        data = aggregate(paths, min_len, req_len)
        kv[problem] = data

    with open('m23sat.pickle', 'wb+') as f:
        pickle.dump(kv, f)


if __name__ == '__main__':
    run()
