import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


with open('qubo_map.pickle', 'rb') as f:
    medians, singles = pickle.load(f)

# Total length is 900k
problems = ["NP", "MC", "MVC", "SP", "M2SAT", "SPP", "GC", "QA", "QK"]

# Plot.
cmap_mod = truncate_colormap('Greens', minval=.3, maxval=.99)


def plot(data):
    fig, axs = plt.subplots(3, 3, figsize=(8, 8.0), constrained_layout=True)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            print(idx)
            im = axs[i][j].imshow(data[idx], cmap=cmap_mod, vmin=-1, vmax=1)
            cbar = axs[i][j].figure.colorbar(im, ax=axs, aspect=60)
            axs[i][j].set_title(problems[idx])
    plt.show()


plot(medians)
plot(singles)
