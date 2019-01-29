'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from sklearn.ensemble.forest import RandomForestClassifier

from data.plot import plot
from data.data import Data
from data.numpy_file import save_np_array, load_np_array
import numpy as np
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc


if __name__ == '__main__':
    '''
    Classify data changing balancing ratio.
    '''

    # Train and test random forests.
    load_path = "../homesite_data/resources/oversampled_normalized_data_ratio_2.5.bin"
    homesite = Data()
    homesite.load_sliptted_data(load_path)
    del homesite.test_x  # Deleted to save memory.

    accuracy_history = []
    precision_history = []
    recall_history = []
    auc_history = []

    for i in range(0, 10):
        if i == 0:
            # Creating classifier.
            clf = RandomForestClassifier(n_estimators = 1, max_features = 100, n_jobs = 4)
        else:
            # Creating classifier.
            clf = RandomForestClassifier(n_estimators = i * 100, max_features = 100, n_jobs = 4)

        # Train classifier.
        print "Training classifier."
        clf.fit(homesite.train_x, homesite.train_y)

        # Test classifier.
        print 'Testing classifier.'
        predicted_labels = clf.predict_proba(homesite.validation_x)[:, 1]

        # Show final results.
        results = confusion_matrix(homesite.validation_y, np.round(predicted_labels))
        accuracy, precision, recall = compute_performance_metrics(results)
        auc = compute_auc(homesite.validation_y, predicted_labels)

        accuracy_history.append(accuracy)
        precision_history.append(precision)
        recall_history.append(recall)
        auc_history.append(auc)

        print 'Saving result.', i * 10
        save_np_array("../homesite_data/results/random_forest_grid_search_accuracy.bin", np.array(accuracy_history))
        save_np_array("../homesite_data/results/random_forest_grid_search_precision.bin", np.array(precision_history))
        save_np_array("../homesite_data/results/random_forest_grid_search_recall.bin", np.array(recall_history))
        save_np_array("../homesite_data/results/random_forest_grid_search_auc.bin", np.array(auc_history))

        del clf