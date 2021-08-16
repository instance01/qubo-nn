import pickle
import numpy as np
import scipy.stats as st
from sklearn.tree import DecisionTreeClassifier


def calc_ci(arr):
    mean = np.mean(arr, axis=0)
    ci = st.t.interval(
        0.95,
        len(arr) - 1,
        loc=np.mean(arr, axis=0),
        scale=st.sem(arr, axis=0)
    )
    range_ = round(mean - ci[0], 8)
    mean = round(mean, 8)
    return mean, range_


node_counts = []
mcs = []
with open('decision_tree2.pickle', 'rb') as f:
    # conf_matrix, report, clf = pickle.load(f)
    results = pickle.load(f)
    for (conf_matrix, report, clf) in results:
        # print(conf_matrix)
        # print(report)
        # print(clf.get_depth())
        # print(clf.tree_.node_count)

        mc = (np.sum(conf_matrix) - np.trace(conf_matrix)) / np.trace(conf_matrix)
        node_counts.append(clf.tree_.node_count)
        mcs.append(mc)

print(calc_ci(node_counts))
print(calc_ci(mcs))
