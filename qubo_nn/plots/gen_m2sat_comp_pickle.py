import os
import glob
import pickle

import numpy as np
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate_single(paths, desired_tag, min_len, len_):
    aggregated = []
    for path in paths:
        path = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if not path:
            continue
        path = path[0]

        data = []
        for event in my_summary_iterator(path):
            if not event.summary.value:
                continue
            tag = event.summary.value[0].tag
            if tag.endswith(desired_tag):
                val = event.summary.value[0].simple_value
                data.append(val)

        # data = smooth(data, 20)[10:-9]
        # data = smooth(data, 60)[30:-29]
        if len(data) == len_ or len(data) == (len_ // 10 + 1):
            print(len(data), desired_tag)
            aggregated.append(data)

    if not aggregated:
        return []
    max_len = max(max(len(x) for x in aggregated), min_len)
    aggregated_ = []
    for i, x in enumerate(aggregated):
        aggregated_.append(
            np.pad(
                x,
                (0, max_len - len(x)),
                mode='constant',
                constant_values=(0, x[-1])
            )
        )
    arr = np.array(aggregated_)
    return arr


def aggregate(paths, min_len, len_):
    fpfn_tot_ratio = aggregate_single(paths, 'Custom/FPFN_TOT_Ratio', min_len, len_)
    r2 = aggregate_single(paths, 'Custom/R2', min_len, len_)
    return fpfn_tot_ratio, r2


def run():
    min_len = 201
    base_path = '../runs/'
    problems = [
        ('m2sat_16x16_5_F', 201),
        ('m2sat_16x16_10_F', 61),
        ('m2sat_16x16_15_F', 51),
        ('m2sat_16x16_20_F', 51),
        ('m2sat_16x16_25_F', 31),
        ('m2sat_16x16_30_F', 31)
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
