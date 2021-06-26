import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# From: qa4_diffnorm_v2 (printed to terminal)


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


x = [1.97, 7.48, 0.8, 8.42, 0.58, 0.79, 1.63, 3.94, 4.02, 3.76, 8.7, 9.55, 9.88, 7.43, 8.96, 7.17, 6.23, 4.36, 3.08, 8.74, 4.62, 7.21, 0.74, 2.95, 4.83, 9.28, 3.74, 9.21, 5.45, 5.91, 8.69, 8.69, 4.75, 5.33, 4.12, 9.63, 4.86, 3.1, 2.46, 2.49, 4.17, 4.56, 5.02, 1.21, 8.14, 8.82, 1.46, 7.93, 10.05, 1.02, 0.48, 10.05, 3.54, 5.22, 6.8, 9.62, 7.31, 2.58, 1.96, 4.68, 9.98, 4.18, 1.85, 9.97, 8.45, 7.41, 1.77, 7.61, 8.92, 7.16, 7.7, 0.11, 9.86, 6.64, 2.49, 9.75, 7.68, 1.01, 8.64, 1.56, 1.16, 2.31, 8.24, 7.03, 0.95, 2.24, 7.85, 5.45, 7.98, 4.1, 4.04, 7.49, 0.76, 8.61, 2.78, 5.87, 6.38, 6.2, 3.49, 3.62, 0.67, 3.36, 7.71, 2.72, 8.22, 3.77, 4.77, 2.83, 3.09, 9.44, 10.02, 7.12, 1.56, 0.53, 8.67, 8.04, 9.52, 7.64, 3.21, 9.16, 0.01, 1.15, 6.8, 9.03, 5.48, 7.92, 3.88, 9.03]  # noqa
y = [2.04, 7.35, 0.82, 8.16, 0.61, 0.82, 1.43, 3.88, 4.08, 3.67, 8.57, 9.59, 9.8, 7.14, 8.98, 7.14, 6.12, 4.29, 2.86, 8.57, 4.49, 7.14, 0.82, 3.06, 4.9, 8.98, 3.67, 9.18, 5.1, 5.92, 8.57, 8.78, 4.49, 5.51, 3.88, 9.39, 4.9, 2.86, 2.45, 2.45, 4.08, 4.69, 4.9, 1.22, 7.96, 8.57, 1.43, 7.76, 9.8, 1.02, 0.41, 9.8, 3.47, 5.1, 6.53, 9.59, 7.35, 2.45, 2.24, 4.49, 9.59, 4.29, 1.84, 9.8, 8.57, 7.55, 2.04, 7.76, 8.98, 7.55, 7.76, 0.2, 10.0, 6.73, 2.24, 9.8, 7.76, 1.02, 8.78, 1.63, 1.02, 2.24, 8.16, 7.14, 1.02, 2.24, 7.96, 5.71, 7.96, 4.08, 3.88, 7.55, 0.82, 8.98, 2.65, 5.92, 6.53, 6.12, 3.47, 3.67, 0.61, 3.27, 7.76, 2.65, 8.16, 3.88, 4.69, 3.06, 3.06, 9.59, 10.0, 7.14, 1.43, 0.61, 8.57, 8.16, 9.8, 7.76, 3.27, 9.18, 0.2, 1.22, 6.73, 8.98, 5.71, 8.16, 3.67, 9.18]  # noqa


# 128 data points is too much to visualize side by side.
x = x[:50]
y = y[:50]


idx = np.arange(len(y))
width = np.min(np.diff(idx)) / 3

fig = plt.figure(figsize=(6, 2))
ax = fig.add_subplot(111)
ax.bar(idx - width + .25, x, width, label="Predicted")
ax.bar(idx + .25, y, width, label="Truth")
plt.ylim(0)
plt.xlim(-1, 50)
plt.legend()
plt.ylabel("Value")
plt.xlabel("QA Parameter")
plt.tight_layout()
fig.savefig('qa_best_99_value.png')
fig.savefig('qa_best_99_value.pdf')
plt.show()

fig = plt.figure(figsize=(6, 2))
diff = np.array(x) - np.array(y)
plt.bar(idx, diff, width)
plt.xlim(-1, 50)
plt.ylabel("Residual")
plt.xlabel("QA Parameter")
plt.tight_layout()
fig.savefig('qa_best_99_residual.png')
fig.savefig('qa_best_99_residual.pdf')
plt.show()
