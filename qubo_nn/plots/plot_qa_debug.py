import pickle

import numpy as np
import matplotlib.pyplot as plt


with open("qa_debug.pickle", "rb") as f:
    data = pickle.load(f)


losses = [d[0] for d in data]
loss90 = np.percentile(losses, 97.5)
loss10 = np.percentile(losses, 2.5)

avg_diff_x = 0
avg_diff_y = 0
x = []  # good ones
y = []  # bad ones
for d in data:
    if d[0] > loss90:
        y.append((d[3][0], d[2][0], d[1][0]))
        avg_diff_y += abs(np.average(d[3][0][:16]) - np.average(d[3][0][16:]))
    elif d[0] < loss10:
        x.append((d[3][0], d[2][0], d[1][0]))
        avg_diff_x += abs(np.average(d[3][0][:16]) - np.average(d[3][0][16:]))

print(avg_diff_x / 1000., avg_diff_y / 1000.)

idx = np.arange(len(y[0][0]))
width = np.min(np.diff(idx)) / 3

print(len(x), len(y))

for i in range(20):
    print(y[i][2])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    ax.bar(idx - width + .25, x[i][0], width, label="Want")
    ax.bar(idx + .25, x[i][0], width, label="Predicted")
    ax.set_title("Good loss")
    ax.axhline(y=np.average(x[i][0][:16]), xmin=0, xmax=.5)
    ax.axhline(y=np.average(x[i][0][16:]), xmin=.5, xmax=1.)

    ax = axs[1]
    ax.bar(idx - width + .25, y[i][0], width, label="Want")
    ax.bar(idx + .25, y[i][1], width, label="Predicted")
    ax.set_title("Bad loss")
    ax.axhline(y=np.average(y[i][0][:16]), xmin=0, xmax=.5, c='orange')
    ax.axhline(y=np.average(y[i][0][16:]), xmin=.5, xmax=1., c='orange')

    plt.legend()
    plt.ylabel("Value")
    plt.xlabel("QA Parameter (QUBO size 16x16)")
    plt.tight_layout()
    plt.show()
