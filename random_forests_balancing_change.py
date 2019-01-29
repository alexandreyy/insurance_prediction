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

#     accuracy_history = []
#     precision_history = []
#     recall_history = []
#     auc_history = []
#     balancing_rate_history = []
#
#     for i in range(35):
#         load_path = "../homesite_data/resources/splitted_data.bin"
#         homesite = Data()
#         homesite.load_sliptted_data(load_path)
#         homesite.z_norm_by_feature()
#         del homesite.test_x  # Deleted to save memory.
#         homesite.train_y = homesite.train_y.flatten()
#         homesite.validation_y = homesite.validation_y.flatten()
#
#         if i > 0:
#             # Balance data.
#             homesite.balance_data_oversampling(ratio = i * 0.1, balance_type = "OverSampler")
#
#         # Creating classifier.
#         clf = RandomForestClassifier(max_features = 100)
#
#         # Train classifier.
#         print "Training classifier."
#         clf.fit(homesite.train_x, homesite.train_y)
#
#         # Test classifier.
#         print 'Testing classifier.'
#         predicted_labels = clf.predict_proba(homesite.validation_x)[:, 1]
#
#         # Show final results.
#         results = confusion_matrix(homesite.validation_y, np.round(predicted_labels))
#         accuracy, precision, recall = compute_performance_metrics(results)
#         auc = compute_auc(homesite.validation_y, predicted_labels)
#
#         accuracy_history.append(accuracy)
#         precision_history.append(precision)
#         recall_history.append(recall)
#         auc_history.append(auc)
#         balancing_rate = np.count_nonzero(homesite.train_y) * 1.0 / len(homesite.train_y)
#         balancing_rate_history.append(balancing_rate)
#
#         print 'Saving result.', i * 0.1
#         save_np_array("../homesite_data/results/random_forest_balancing_accuracy.bin", np.array(accuracy_history))
#         save_np_array("../homesite_data/results/random_forest_balancing_precision.bin", np.array(precision_history))
#         save_np_array("../homesite_data/results/random_forest_balancing_recall.bin", np.array(recall_history))
#         save_np_array("../homesite_data/results/random_forest_balancing_auc.bin", np.array(auc_history))
#         save_np_array("../homesite_data/results/random_forest_balancing_rate.bin", np.array(balancing_rate_history))
#
#         del homesite
#         del clf

    accuracy_history = load_np_array("../homesite_data/results/random_forest_balancing_accuracy.bin")
    precision_history = load_np_array("../homesite_data/results/random_forest_balancing_precision.bin")
    recall_history = load_np_array("../homesite_data/results/random_forest_balancing_recall.bin")
    auc_history = load_np_array("../homesite_data/results/random_forest_balancing_auc.bin")
    balancing_rate_history = load_np_array("../homesite_data/results/random_forest_balancing_rate.bin")

#     for accuracy, precision, recall, auc, balancing_rate in zip(accuracy_history, precision_history, recall_history, auc_history, balancing_rate_history):
#         print accuracy, precision, recall, auc, balancing_rate


    plot("../homesite_data/results/random_forest_balacing.png", [recall_history, auc_history], \
         ["sensitividade ", "AUC"], "taxa de balanceamento", "metricas", 'center right', \
         x = np.linspace(0, len(recall_history) / 10, num = len(recall_history), endpoint = True))
