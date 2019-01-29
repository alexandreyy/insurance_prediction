'''
Created on 26/11/2015

@author: Alexandre Yukio Yamashita
'''

from classifiers.neural_network import NeuralNetwork
from data.data import Data
from data.numpy_file import save_np_array, load_np_array
import numpy as np
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
from data.plot import plot


if __name__ == '__main__':
    '''
    Compare normalization vs non-normalization.
    '''

#     # Train neural network with non normalized data.
#     cost_path = "results/ann_non_normalized_cost_history.bin"
#     oversampled_path = "resources/oversampled_data_ratio_2.bin"
#     homesite_data = Data()
#     homesite_data.load_sliptted_data(oversampled_path, one_hot = True)
#     clf = NeuralNetwork(input_units = 644, hidden_units = 50, output_units = 2, \
#                         lr = 0.0000005, lamb = 0.)
#     cost_history = clf.fit(homesite_data, batch_size = 128, \
#                            max_iterations = 100, save_interval = 10, \
#                            path = "classifiers_data/ann_weights.bin",
#                            return_cost = True)
#
#     # Save cost and accuracy history
#     save_np_array(cost_path, cost_history)

#     # Train neural network with non normalized data.
#     cost_path = "results/ann_normalized_cost_history.bin"
#     oversampled_path = "resources/oversampled_normalized_data_ratio_2.bin"
#     homesite_data = Data()
#     homesite_data.load_sliptted_data(oversampled_path, one_hot = True)
#     clf = NeuralNetwork(input_units = 644, hidden_units = 50, output_units = 2, \
#                         lr = 0.005, lamb = 0.)
#     cost_history = clf.fit(homesite_data, batch_size = 128, \
#                            max_iterations = 100, save_interval = 10, \
#                            path = "classifiers_data/ann_weights.bin",
#                            return_cost = True)
#
#     # Save cost and accuracy history
#     save_np_array(cost_path, cost_history)

#     # Test neural network.
#     oversampled_path = "resources/oversampled_normalized_data_ratio_2.bin"
#     homesite_data = Data()
#     homesite_data.load_sliptted_data(oversampled_path, one_hot = True)
#     clf = NeuralNetwork(path = "classifiers_data/ann_weights.bin", \
#                         lr = 0.005, lamb = 0.)
#     prob_predicted_labels = clf.predict(homesite_data.validation_x)
#     predicted_labels = np.argmax(prob_predicted_labels, axis = 1)
#     validation_labels = np.argmax(homesite_data.validation_y, axis = 1)
#
#     # Show final results.
#     results = confusion_matrix(validation_labels, predicted_labels)
#     accuracy, precision, recall = compute_performance_metrics(results)
#     auc = compute_auc(validation_labels, prob_predicted_labels[:, 1])

    # Save plot.
    non_normalized_cost = load_np_array("results/ann_non_normalized_cost_history.bin")
    normalized_cost = load_np_array("results/ann_normalized_cost_history.bin")

    plot("results/normalization_vs_non_normalization.png", [normalized_cost, non_normalized_cost], \
         ["com normalizacao", "sem normalizacao"], "iteracoes", "custo", 'center right')
