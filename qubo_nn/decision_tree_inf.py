import pickle
from sklearn.tree import DecisionTreeClassifier


with open('decision_tree2.pickle', 'rb') as f:
    conf_matrix, report, clf = pickle.load(f)
print(conf_matrix)
print(report)
print(clf.get_depth())
print(clf.tree_.node_count)

import pdb; pdb.set_trace()
