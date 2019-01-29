'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np
from data import Data


def split_data(X, y, ratio = 0.70):
    '''
    Split data.
    '''

    labels = np.unique(y)
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None

    for label in labels:
        data_by_label = X[np.where(np.equal(label, y)), :]
        data_by_label = data_by_label.reshape(data_by_label.shape[1], data_by_label.shape[2])
        half_index = int(len(data_by_label) * ratio + 0.5)

        if train_X is None:
            train_X = data_by_label[0: half_index, :]
            test_X = data_by_label[half_index:, :]
            train_Y = np.ones((len(train_X), 1)) * label
            test_Y = np.ones((len(test_X), 1)) * label
        else:
            data_train = data_by_label[0: half_index, :]
            data_test = data_by_label[half_index:, :]
            train_X = np.vstack((train_X, data_train))
            test_X = np.vstack((test_X, data_test))
            train_Y = np.vstack((train_Y, np.ones((len(data_train), 1)) * label))
            test_Y = np.vstack((test_Y, np.ones((len(data_test), 1)) * label))

        # Shuffle data.
        indexes_train = range(len(train_X))
        np.random.shuffle(indexes_train)
        train_X = train_X[indexes_train, :]
        train_Y = train_Y[indexes_train, :]
        indexes_test = range(len(test_X))
        np.random.shuffle(indexes_test)
        test_X = test_X[indexes_test, :]
        test_Y = test_Y[indexes_test, :]

    return train_X, train_Y.astype(bool), test_X, test_Y.astype(bool)


if __name__ == '__main__':
    '''
    Split data.
    '''

    load_path = "../../homesite_data/resources/parsed_data.bin"
    save_path = "../../homesite_data/resources/splitted_data.bin"
    homesite = Data()
    homesite.load_parsed_data(load_path)
    print homesite.train_x.shape, homesite.train_y.shape, homesite.test_x.shape
#
#     # Split data.
#     train_X, train_Y, validation_X, validation_Y = split_data(homesite.train_x, homesite.train_y, 0.7)
#     homesite.train_x = train_X
#     homesite.train_y = train_Y
#     homesite.validation_x = validation_X
#     homesite.validation_y = validation_Y
#     homesite.save_sliptted_data(save_path)

#     load_path = "../../homesite_data/resources/splitted_data.bin"
#     homesite = Data()
#     homesite.load_sliptted_data(load_path)
#     print homesite.train_y.shape, homesite.validation_x.shape
#     y = np.count_nonzero(homesite.validation_y) + np.count_nonzero(homesite.train_y)
#     t_size = len(homesite.validation_y) + len(homesite.train_y)
#     print y, t_size, y * 1.0 / t_size, (t_size - y) * 1.0 / t_size
