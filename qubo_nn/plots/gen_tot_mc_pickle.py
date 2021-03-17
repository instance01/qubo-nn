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


def aggregate(base_path, id_):
    paths = glob.glob('%s*-%s' % (base_path, id_))

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

        if len(data) < 40:
            continue
        # data = smooth(data, 20)[10:-9]
        # data = smooth(data, 60)[30:-29]
        data = data[:40]
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

    with open('tot_misclassifications_%s.pickle' % id_, 'wb+') as f:
        pickle.dump(arr, f)

    return arr


def run():
    aggregate('../runs/', '27_scramble_100k')
    aggregate('../runs2/', '18_lr2_leaky')
    aggregate('../runs2/', '23')


if __name__ == '__main__':
    run()
