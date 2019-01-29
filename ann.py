'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from sklearn.ensemble.forest import RandomForestClassifier

from classifiers.neural_network import NeuralNetwork
from data.data import Data
from data.numpy_file import save_np_array, load_np_array
from data.plot import plot
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
import numpy as np
import math

if __name__ == '__main__':
    '''
    Classify data changing balancing ratio.
    '''

    # Train and test random forests.
    # load_path = "../homesite_data/resources/oversampled_normalized_data_ratio_2.5.bin"
    load_path = "../homesite_data/resources/oversampled_normalized_data_ratio_2.bin"
    homesite = Data()
    homesite.load_sliptted_data(load_path, one_hot = True)
    del homesite.test_x  # Deleted to save memory.

    accuracy_history = []
    precision_history = []
    recall_history = []
    auc_history = []

#     for i in range(0, 10):
#         # Creating classifier.
#         clf = NeuralNetwork(input_units = 644, hidden_units = 50, output_units = 2, \
#                             lr = 0.00005, lamb = 0.000000005 * math.pow(10, i))
#
#         # Train classifier.
#         print "Training classifier."
#
#         clf.fit(homesite, batch_size = 128,
#             max_iterations = 500, save_interval = 500,
#             path = "../homesite_data/resources/ann_weights.bin")
#
#         # Test classifier.
#         print 'Testing classifier.'
#         predicted_labels = clf.predict_proba(homesite.validation_x)[:, 1]
#
#         # Show final results.
#         results = confusion_matrix(np.argmax(homesite.validation_y, axis = 1), np.round(predicted_labels))
#         accuracy, precision, recall = compute_performance_metrics(results)
#         auc = compute_auc(np.argmax(homesite.validation_y, axis = 1), predicted_labels)
#
#         accuracy_history.append(accuracy)
#         precision_history.append(precision)
#         recall_history.append(recall)
#         auc_history.append(auc)
#
#         print 'Saving result.', i * 10
#         save_np_array("../homesite_data/results/ann_grid_search_accuracy.bin", np.array(accuracy_history))
#         save_np_array("../homesite_data/results/ann_grid_search_precision.bin", np.array(precision_history))
#         save_np_array("../homesite_data/results/ann_grid_search_recall.bin", np.array(recall_history))
#         save_np_array("../homesite_data/results/ann_grid_search_auc.bin", np.array(auc_history))
#
#         del clf

    for i in range(0, 10):
        # Creating classifier.
        if i == 0:
            a = 1
        else:
            a = i * 10

        clf = NeuralNetwork(input_units = 644, hidden_units = a, output_units = 2, \
                   lr = 0.05, lamb = 0)

        # Train classifier.
        print "Training classifier."

        clf.fit(homesite, batch_size = 128,
            max_iterations = 500, save_interval = 500,
            path = "../homesite_data/resources/ann_weights.bin")

        # Test classifier.
        print 'Testing classifier.'
        predicted_labels = clf.predict_proba(homesite.validation_x)[:, 1]

        # Show final results.
        results = confusion_matrix(np.argmax(homesite.validation_y, axis = 1), np.round(predicted_labels))
        accuracy, precision, recall = compute_performance_metrics(results)
        auc = compute_auc(np.argmax(homesite.validation_y, axis = 1), predicted_labels)

        accuracy_history.append(accuracy)
        precision_history.append(precision)
        recall_history.append(recall)
        auc_history.append(auc)

        print 'Saving result.', i * 10
        save_np_array("../homesite_data/results/ann_grid_search_2_accuracy.bin", np.array(accuracy_history))
        save_np_array("../homesite_data/results/ann_grid_search_2_precision.bin", np.array(precision_history))
        save_np_array("../homesite_data/results/ann_grid_search_2_recall.bin", np.array(recall_history))
        save_np_array("../homesite_data/results/ann_grid_search_2_auc.bin", np.array(auc_history))

        del clf
