'''
Created on 27/11/2015

@author: Alexandre Yukio Yamashita
'''

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


def plot(path, data, labels, xlabel, ylabel, loc = 'lower right', x = None):
    '''
    Plot data.
    '''

    if x is None:
        x = range(len(data[0]))

    for d in data:
        f = interp1d(x, d, kind = 'slinear')
        n_x = np.linspace(np.min(x), np.max(x), num = 100, endpoint = True)
        plt.plot(n_x, f(n_x))

        # plt.plot(x, d)

    plt.legend(labels, loc = loc)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.show()
    plt.savefig(path, bbox_inches = 'tight')
