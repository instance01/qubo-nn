import pickle

import numpy as np
import scipy.stats as st
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from lib import cmap_mod


large_matrix = [
    [-4., 4., 4., 4., 2., 0., 0., 0., 2., 0., 0., 0.],
    [ 4.,-4., 4., 4., 0., 2., 0., 0., 0., 2., 0., 0.],
    [ 4., 4.,-4., 4., 0., 0., 2., 0., 0., 0., 2., 0.],
    [ 4., 4., 4.,-4., 0., 0., 0., 2., 0., 0., 0., 2.],
    [ 2., 0., 0., 0.,-4., 4., 4., 4., 2., 0., 0., 0.],
    [ 0., 2., 0., 0., 4.,-4., 4., 4., 0., 2., 0., 0.],
    [ 0., 0., 2., 0., 4., 4.,-4., 4., 0., 0., 2., 0.],
    [ 0., 0., 0., 2., 4., 4., 4.,-4., 0., 0., 0., 2.],
    [ 2., 0., 0., 0., 2., 0., 0., 0.,-4., 4., 4., 4.],
    [ 0., 2., 0., 0., 0., 2., 0., 0., 4.,-4., 4., 4.],
    [ 0., 0., 2., 0., 0., 0., 2., 0., 4., 4.,-4., 4.],
    [ 0., 0., 0., 2., 0., 0., 0., 2., 4., 4., 4.,-4.]
]

small_matrix = [
    [-4., 4., 4., 2., 0., 0., 2., 0., 0., 0., 0., 0.],
    [ 4.,-4., 4., 0., 2., 0., 0., 2., 0., 0., 0., 0.],
    [ 4., 4.,-4., 0., 0., 2., 0., 0., 2., 0., 0., 0.],
    [ 2., 0., 0.,-4., 4., 4., 2., 0., 0., 0., 0., 0.],
    [ 0., 2., 0., 4.,-4., 4., 0., 2., 0., 0., 0., 0.],
    [ 0., 0., 2., 4., 4.,-4., 0., 0., 2., 0., 0., 0.],
    [ 2., 0., 0., 2., 0., 0.,-4., 4., 4., 0., 0., 0.],
    [ 0., 2., 0., 0., 2., 0., 4.,-4., 4., 0., 0., 0.],
    [ 0., 0., 2., 0., 0., 2., 4., 4.,-4., 0., 0., 0.],
    [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
]


def plot(matrix, id_):
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.tight_layout()
    ax.imshow(matrix, cmap=cmap_mod)#, vmin=0, vmax=1)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('gc_matrix_%s.png' % id_)
    plt.savefig('gc_matrix_%s.pdf' % id_)
    plt.show()


plot(large_matrix, 'large')
plot(small_matrix, 'small')
