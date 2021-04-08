import os
import glob

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


def aggregate_single(paths, desired_tag, min_len, cutoff=None, req_len=None):
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
        if len(data) == 0:
            continue
        if req_len is not None:
            if len(data) <= req_len:
                continue
        print(len(data), data[-1])
        if cutoff is not None:
            data = data[:cutoff]
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
