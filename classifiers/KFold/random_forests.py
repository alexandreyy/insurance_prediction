'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from data.data import Data
from data.numpy_file import save_np_array
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier
from unbalanced_dataset.over_sampling import OverSampler

def plot_roc(mean_fpr, mean_tpr, mean_auc):
    plt.plot(mean_fpr, mean_tpr, 'k--', label = 'Mean ROC (area = %0.2f)' % mean_auc, lw = 2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = "lower right")
    plt.savefig("../../results/random_forests/roc.png", bbox_inches = 'tight')


if __name__ == '__main__':
    '''
    Classify data
    '''

    accuracy_history = []
    precision_history = []
    recall_history = []
    auc_history = []
    confusion_matrix_history = np.array([])
    confusion_matrix_history.shape = (2, 2, 0)

    path = "../../../homesite_data/resources/parsed_data.bin"
    homesite = Data()
    homesite.load_parsed_data(path)
    homesite.z_norm_train_test_by_feature()
    del homesite.test_x  # Deleted to save memory.

#     reduced_range = range(0, 100)
#     homesite.train_x = homesite.train_x[reduced_range]
#     homesite.train_y = homesite.train_y[reduced_range]

    # Creating classifier.
    mean_acc = 0.0
    mean_recall = 0.0
    mean_precision = 0.0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    c = 300

    cvs = StratifiedKFold(homesite.train_y, n_folds = 5)
    clf = RandomForestClassifier(n_estimators = c, max_features = 100, n_jobs = 4)

    # Train classifier.
    print "\nTraining classifier param %d" % c

    for i, (train, test) in enumerate(cvs):
        sm = OverSampler(verbose = False, ratio = 2.5)
        train_oversampled_x, train_oversampled_train_y = sm.fit_transform(homesite.train_x[train], homesite.train_y[train])
        probas_ = clf.fit(train_oversampled_x, train_oversampled_train_y).predict_proba(homesite.train_x[test])

        fpr, tpr, thresholds = roc_curve(homesite.train_y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = compute_auc(homesite.train_y[test], probas_[:, 1])
        fold_cm = confusion_matrix(homesite.train_y[test], np.round(probas_)[:, 1])
        confusion_matrix_history = np.dstack((confusion_matrix_history, fold_cm))

        accuracy, precision, recall = compute_performance_metrics(fold_cm)
        mean_acc += accuracy
        mean_recall += recall
        mean_precision += precision

        accuracy_history.append(accuracy)
        precision_history.append(precision)
        recall_history.append(recall)
        auc_history.append(roc_auc)

        save_np_array("../../results/random_forests/rf_accuracy_" + str(c) + ".bin", np.array(accuracy_history))
        save_np_array("../../results/random_forests/rf_precision_" + str(c) + ".bin", np.array(precision_history))
        save_np_array("../../results/random_forests/rf_recall_" + str(c) + ".bin", np.array(recall_history))
        save_np_array("../../results/random_forests/rf_auc_" + str(c) + ".bin", np.array(auc_history))
        save_np_array("../../results/random_forests/rf_confusion_matrix_" + str(c) + ".bin", np.array(confusion_matrix_history))
        plt.plot(fpr, tpr, lw = 1, label = 'ROC fold %d (area = %0.2f)' % (i, roc_auc))

    mean_acc /= len(cvs)
    mean_recall /= len(cvs)
    mean_precision /= len(cvs)
    mean_tpr /= len(cvs)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plot_roc(mean_fpr, mean_tpr, mean_auc)
