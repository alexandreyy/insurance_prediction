'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

import math
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from classifiers.neural_network import NeuralNetwork
from data.data import Data
from data.numpy_file import save_np_array, load_np_array
from data.plot import plot
import numpy as np
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc


if __name__ == '__main__':
    '''
    Classify data changing balancing ratio.
    '''

    # Train and test random forests.
    # load_path = "../homesite_data/resources/oversampled_normalized_data_ratio_2.5.bin"
    load_path = "../homesite_data/resources/oversampled_normalized_data_ratio_2.bin"
    homesite = Data()
    homesite.load_sliptted_data(load_path)
    del homesite.test_x  # Deleted to save memory.

    clf_ann = NeuralNetwork(path = "../homesite_data/ann_weights.bin", lr = 0.00005, \
                        lamb = 0)
    train_output_ann = clf_ann.get_hidden_output(homesite.train_x)
    validation_output_ann = clf_ann.get_hidden_output(homesite.validation_x)
    # train_output_ann = np.hstack((train_output_ann, homesite.train_x))
    # validation_output_ann = np.hstack((validation_output_ann, homesite.validation_x))

    for c in range(1, 10):
        # Train classifier.
        print "Training classifier."
        clf = AdaBoostClassifier(n_estimators = 1 + 100 * c)
        clf.fit(train_output_ann, homesite.train_y)

        # Test classifier.
        print 'Testing classifier.'
        predicted_labels = clf.predict_proba(validation_output_ann)[:, 1]

        # Show final results.
        results = confusion_matrix(homesite.validation_y, np.round(predicted_labels))
        accuracy, precision, recall = compute_performance_metrics(results)
        auc = compute_auc(homesite.validation_y, predicted_labels)
