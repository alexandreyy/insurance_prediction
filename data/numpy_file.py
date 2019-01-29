'''
Created on 26/11/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np


def save_np_array(path, np_array):
    '''
    Save numpy array.
    '''

    f = file(path, "wb")
    np.save(f, np_array)
    f.close()


def load_np_array(path):
    '''
    Load numpy array.
    '''

    f = file(path, "rb")
    np_array = np.load(f)
    f.close()

    return np_array