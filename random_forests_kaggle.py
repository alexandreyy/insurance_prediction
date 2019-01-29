# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 19:59:26 2015

@author: celsokakihara
"""
'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from sklearn.ensemble.forest import RandomForestClassifier
from unbalanced_dataset.over_sampling import OverSampler

from data.data import Data
from data.numpy_file import save_np_array, load_np_array
from data.plot import plot
import numpy as np
import pandas as pd
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc


if __name__ == '__main__':
    '''
    Classify data changing balancing ratio.
    '''

    # Train and test random forests.
    path = "../homesite_data/resources/parsed_data.bin"
    homesite = Data()
    homesite.load_parsed_data(path)
    homesite.z_norm_train_test_by_feature()
    sm = OverSampler(verbose = False, ratio = 2.5)
    homesite.train_x, homesite.train_y = sm.fit_transform(homesite.train_x, homesite.train_y)

    clf = RandomForestClassifier(n_estimators = 300, max_features = 100, n_jobs = 4)

    # Train classifier.
    print "Training classifier."
    clf.fit(homesite.train_x, homesite.train_y)
    predicted_labels = clf.predict_proba(homesite.test_x)[:, 1]
    sample = pd.read_csv('../input/sample_submission.csv')
    sample.QuoteConversion_Flag = predicted_labels
    sample.to_csv('rfc_300.csv', index = False)

