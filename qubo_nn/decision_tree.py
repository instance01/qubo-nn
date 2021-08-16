import pickle

import pyxis as px
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from qubo_nn.config import Config
from qubo_nn.data import LMDBDataLoader


cfg = Config().get_cfg('100_genX')
cfg["use_big"] = False
lmdb_loader = LMDBDataLoader(cfg)
loader =lmdb_loader.train_data_loader
X = np.array([])
y = np.array([])
for i, item in enumerate(loader):
    if i % 10 == 0:
        print(i)
    if X.shape[0] == 0:
        X = item[0].detach().numpy().reshape((500, 4096))
        y = item[1].detach().numpy()
    if item[0].shape[0] == 500:
        X = np.concatenate([X, item[0].detach().numpy().reshape((500, 4096))])
        y = np.concatenate([y, item[1].detach().numpy()])

print(X.shape, y.shape)

results = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # test size was .25 per default.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results.append((conf_matrix, report, clf))
with open('decision_tree2.pickle', 'wb+') as f:
    pickle.dump(results, f)
