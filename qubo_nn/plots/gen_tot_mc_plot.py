import os
import sys
import glob
import pickle
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate_multi(arr1):
    fig, ax = plt.subplots()

    def sub_plot(arr, col):
        mean = np.mean(arr, axis=0)
        x = np.arange(mean.shape[0])
        # TODO Use scipy here.. no idea if this is correct.
        ci = 1.96 * np.std(arr, axis=0) / np.mean(arr, axis=0)

        ax.plot(x, mean, color=col)
        # ax.fill_between(x, (mean - ci), (mean + ci), color=col, alpha=.1)

    sub_plot(arr1, 'c')
    plt.ylabel("Misclassification rate")
    plt.xlabel("Epoch")
    plt.show()
    fig.savefig('tot_mc.png')


def aggregate(paths):
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
            if tag.endswith('Total_Misclassifications'):
                # val = float(tf.make_ndarray(event.summary.value[0].tensor))
                val = event.summary.value[0].simple_value
                # print(val)
                data.append(val)

        if len(data) < 30:
            continue
        # data = smooth(data, 20)[10:-9]
        # data = smooth(data, 60)[30:-29]
        data = data[:30]
        print(len(data))
        aggregated.append(data)

    if not aggregated:
        return []
    max_len = max(len(x) for x in aggregated)
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


def run():
    plot = True
    if not plot:
        # paths = glob.glob('../runs/*-18_lr2_leaky')
        paths = glob.glob('../runs/*-23')
        print(len(paths))
        arr1 = aggregate(paths)
        with open('tot_mc.pickle', 'wb+') as f:
            pickle.dump(arr1, f)
    else:
        with open('tot_mc.pickle', 'rb') as f:
            arr1 = pickle.load(f)
        aggregate_multi(arr1)


if __name__ == '__main__':
    run()
