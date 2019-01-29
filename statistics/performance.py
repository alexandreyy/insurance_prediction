'''
Created on 16/11/2015

@author: Alexandre Yukio Yamashita
'''

from sklearn import metrics


def compute_performance_metrics(confusion_matrix):
    '''
    Compute performance metrics.
    '''
    accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0]) * 1.0 / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1])
    precision = confusion_matrix[1, 1] * 1.0 / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
    recall = confusion_matrix[1, 1] * 1.0 / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    sensitivity = confusion_matrix[1, 1] * 1.0 / (confusion_matrix[1, 1] + confusion_matrix[0, 1])

    print "Accuracy | %f" % accuracy
    print "Precision | %f" % precision
    print "Recall | %f" % recall
    print "Sensitivity | %f" % sensitivity

    return accuracy, precision, recall, sensitivity


def compute_roc_curve(true_labels, predicted_labels):
    '''
    Compute roc curve.
    '''
    fpr, tpr, thresholds = metrics.roc_curve(true_labels.flatten(), predicted_labels.flatten(), pos_label = 1)

    return fpr, tpr, thresholds


def compute_auc(true_labels, predicted_labels):
    '''
    Compute area under the curve.
    '''
    fpr, tpr, _ = compute_roc_curve(true_labels, predicted_labels)
    auc = metrics.auc(fpr, tpr)
    print "AUC | %f" % auc

    return auc
