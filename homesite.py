'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from sklearn import svm
from sklearn.cross_decomposition.pls_ import PLSRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.tree.tree import DecisionTreeClassifier

from data.data import Data
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
import numpy as np


if __name__ == '__main__':
    '''
    Classify data.
    '''

    oversampled_path = "resources/oversampled_normalized_data_ratio_2.bin"
    homesite = Data()
    homesite.load_sliptted_data(oversampled_path)
    del homesite.test_x  # Deleted to save memory.
    print homesite.train_x.shape

    # Creating classifier.
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier(max_features = 100)
    # clf = AdaBoostClassifier(n_estimators = 10)
    # clf = svm.SVC(gamma = 0.00005)
    # clf = RandomForestClassifier()
    # clf = MultiplePLS(n_classifiers = 10, n_samples = 5000, n_positive_samples = 2500, threshold = 0.9, acc = 0.999)
    # clf = svm.LinearSVC()

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
