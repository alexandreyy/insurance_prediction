'''
Created on 30/11/2015

@author: Alexandre Yukio Yamashita
'''
from data.numpy_file import load_np_array
import numpy as np
from data.plot import plot

# Plot grid search.
C = [16, 32, 64, 128, 256, 512]
auc = []
recall = []

for c in C:
    recall_history = load_np_array("results/adaboost/ada_recall_" + str(c) + ".bin")
    recall_history = recall_history[len(recall_history) - 5:]
    accuracy_history = load_np_array("results/adaboost/ada_accuracy_" + str(c) + ".bin")
    accuracy_history = recall_history[len(accuracy_history) - 5:]
    precision_history = load_np_array("results/adaboost/ada_precision_" + str(c) + ".bin")
    precision_history = precision_history[len(precision_history) - 5:]
    auc_history = load_np_array("results/adaboost/ada_auc_" + str(c) + ".bin")
    auc_history = auc_history[len(auc_history) - 5:]
    auc.append(np.mean(auc_history))
    recall.append(np.mean(recall_history))

recall = np.array(recall)
auc = np.array(auc)

plot("results/adaboost/adaboost_grid_search.png", [recall, auc], \
    ["sensitividade ", "AUC"], "numero de estimadores", "metricas", 'center right', \
    x = np.array(C))

# Show confusion matrix for best c.
confusion_matrix_history = load_np_array("results/adaboost/rfc_folds_confusion_256.bin")
confusion_matrix_history = confusion_matrix_history[:, :, len(confusion_matrix_history[0, 0]) - 5:]
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

# Show metrics for best c.
c = 256
recall_history = load_np_array("results/adaboost/ada_recall_" + str(c) + ".bin")
recall_history = recall_history[len(recall_history) - 5:]
accuracy_history = load_np_array("results/adaboost/ada_accuracy_" + str(c) + ".bin")
accuracy_history = recall_history[len(accuracy_history) - 5:]
precision_history = load_np_array("results/adaboost/ada_precision_" + str(c) + ".bin")
precision_history = precision_history[len(precision_history) - 5:]
auc_history = load_np_array("results/adaboost/ada_auc_" + str(c) + ".bin")
auc_history = auc_history[len(auc_history) - 5:]

print "\nAccuracy: %.2f%% +- %.2f%%" % (np.mean(accuracy_history) * 100, np.std(accuracy_history) * 100)
print "Precision: %.2f%% +- %.2f%%" % (np.mean(precision_history) * 100, np.std(precision_history) * 100)
print "Recall/Sensitivity: %.2f%% +- %.2f%%" % (np.mean(recall_history) * 100, np.std(recall_history) * 100)
print "AUC: %.2f%% +- %.2f%%" % (np.mean(auc_history) * 100, np.std(auc_history) * 100)

