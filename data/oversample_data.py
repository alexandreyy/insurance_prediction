'''
Created on 25/11/2015

@author: Alexandre Yukio Yamashita
'''

from data import Data


if __name__ == '__main__':
    '''
    Save oversampled and normalized data.
    '''

    load_path = "../resources/splitted_data.bin"
    oversampled_path = "../resources/oversampled_data_ratio_2.bin"
    homesite = Data()
    homesite.load_sliptted_data(load_path)
    # homesite.z_norm_by_feature()
    homesite.train_y = homesite.train_y.flatten()
    homesite.validation_y = homesite.validation_y.flatten()

    # Balance data.
    homesite.balance_data_oversampling(ratio = 2, balance_type = "OverSampler")

    # Save oversampled data.
    homesite.save_sliptted_data(oversampled_path)
