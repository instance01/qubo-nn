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


def aggregate(base_paths, id_, req_len, confusion_matrix_req=True):
    paths = []
    for base_path in base_paths:
        paths.extend(glob.glob('%s*-%s' % (base_path, id_)))

    aggregated = []
    aggregated_confusion_matrices = []
    for path in paths:
        tf_path = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if not tf_path:
            continue
        tf_path = tf_path[0]

        data = []
        for event in my_summary_iterator(tf_path):
            if not event.summary.value:
                continue
            tag = event.summary.value[0].tag
            if tag.endswith('Total_Misclassifications'):
                # val = float(tf.make_ndarray(event.summary.value[0].tensor))
                val = event.summary.value[0].simple_value
                # print(val)
                data.append(val)

        print(len(data))

        if confusion_matrix_req:
            # Also, let's get the confusion matrix:
            path = glob.glob(os.path.join(path, "confusion_matrix_data.pickle"))
            if not path:
                continue  # We now require a confusion matrix.
            path = path[0]
            with open(path, "rb") as f:
                matrix = pickle.load(f)
            aggregated_confusion_matrices.append(matrix)

        if len(data) < req_len:
            continue
        # data = smooth(data, 20)[10:-9]
        # data = smooth(data, 60)[30:-29]
        data = data[:200]
        aggregated.append(data)

    print("Confusion matrices", len(aggregated_confusion_matrices))
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

    with open('tot_mc_100_genX_%s.pickle' % id_, 'wb+') as f:
        pickle.dump((arr, aggregated_confusion_matrices), f)

    return arr


def run():
    aggregate(['../runs/'], '100_genX', 50)
    aggregate(['../runs/'], '100_genX_2', 15, False)


if __name__ == '__main__':
    run()
