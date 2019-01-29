'''
Created on 30/11/2015

@author: Alexandre Yukio Yamashita
'''
from data.numpy_file import load_np_array
from data.plot import plot
import numpy as np

# # Plot results.
# accuracy_history = load_np_array("../homesite_data/results/random_forest_grid_search_accuracy.bin")
# precision_history = load_np_array("../homesite_data/results/random_forest_grid_search_precision.bin")
# recall_history = load_np_array("../homesite_data/results/random_forest_grid_search_recall.bin")
# auc_history = load_np_array("../homesite_data/results/random_forest_grid_search_auc.bin")
#
# for accuracy, precision, recall, auc in zip(accuracy_history, precision_history, recall_history, auc_history):
#     print accuracy, precision, recall, auc
#
# plot("../homesite_data/results/random_forest_grid_search.png", [recall_history, auc_history], \
#      ["sensitividade ", "AUC"], "numero de arvores", "metricas", 'center right', \
#         x = np.linspace(1, len(recall_history) * 10, num = len(recall_history), endpoint = True))

c = 50
accuracy_history = load_np_array("results/ann/ann_accuracy_" + str(c) + ".bin")
precision_history = load_np_array("results/ann/ann_precision_" + str(c) + ".bin")
recall_history = load_np_array("results/ann/ann_recall_" + str(c) + ".bin")
auc_history = load_np_array("results/ann/ann_auc_" + str(c) + ".bin")
confusion_matrix_history = load_np_array("results/ann/ann_confusion_matrix_" + str(c) + ".bin")

# Show confusion matrix for best c.
confusion_matrix_mean = np.zeros(4)
confusion_matrix_std = np.zeros(4)
confusion_matrix_mean[0] = np.mean(confusion_matrix_history[0, 0, :] * 100.0 / (confusion_matrix_history[0, 0, :] + confusion_matrix_history[0, 1, :]))
confusion_matrix_mean[1] = np.mean(confusion_matrix_history[0, 1, :] * 100.0 / (confusion_matrix_history[0, 0, :] + confusion_matrix_history[0, 1, :]))
confusion_matrix_mean[2] = np.mean(confusion_matrix_history[1, 0, :] * 100.0 / (confusion_matrix_history[1, 0, :] + confusion_matrix_history[1, 1, :]))
confusion_matrix_mean[3] = np.mean(confusion_matrix_history[1, 1, :] * 100.0 / (confusion_matrix_history[1, 0, :] + confusion_matrix_history[1, 1, :]))
confusion_matrix_std[0] = np.std(confusion_matrix_history[0, 0, :] * 100.0 / (confusion_matrix_history[0, 0, :] + confusion_matrix_history[0, 1, :]))
confusion_matrix_std[1] = np.std(confusion_matrix_history[0, 1, :] * 100.0 / (confusion_matrix_history[0, 0, :] + confusion_matrix_history[0, 1, :]))
confusion_matrix_std[2] = np.std(confusion_matrix_history[1, 0, :] * 100.0 / (confusion_matrix_history[1, 0, :] + confusion_matrix_history[1, 1, :]))
confusion_matrix_std[3] = np.std(confusion_matrix_history[1, 1, :] * 100.0 / (confusion_matrix_history[1, 0, :] + confusion_matrix_history[1, 1, :]))

print "Confusion matrix:"
print "(real , predicted) | count"
print "(%s , %s) | %.2f%%, +- %.2f%%" % ("False", "False", confusion_matrix_mean[0], confusion_matrix_std[0])
print "(%s , %s) | %.2f%%, +- %.2f%%" % ("False", "True", confusion_matrix_mean[1], confusion_matrix_std[1])
print "(%s , %s) | %.2f%%, +- %.2f%%" % ("True", "False", confusion_matrix_mean[2], confusion_matrix_std[2])
print "(%s , %s) | %.2f%%, +- %.2f%%" % ("True", "True", confusion_matrix_mean[3], confusion_matrix_std[3])

print "\nAccuracy: %.2f%% +- %.2f%%" % (np.mean(accuracy_history) * 100, np.std(accuracy_history) * 100)
print "Precision: %.2f%% +- %.2f%%" % (np.mean(precision_history) * 100, np.std(precision_history) * 100)
print "Recall/Sensitivity: %.2f%% +- %.2f%%" % (np.mean(recall_history) * 100, np.std(recall_history) * 100)
print "AUC: %.2f%% +- %.2f%%" % (np.mean(auc_history) * 100, np.std(auc_history) * 100)
