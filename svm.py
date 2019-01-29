'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

import csv
from sklearn import svm
from data.data import Data
from data.numpy_file import save_np_array
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
import numpy as np


if __name__ == '__main__':
    '''
    Classify data 
    '''

    results_f = open("svm_results.csv", 'w')
    wr = csv.writer(results_f)
    
    accuracy_history = []
    precision_history = []
    recall_history = []
    auc_history = []
    
    oversampled_path = "resources/oversampled_data_ratio_2.bin"
    homesite = Data()
    homesite.load_sliptted_data(oversampled_path)
    homesite.z_norm_by_feature()
    del homesite.test_x 
    # Deleted to save memory.
    homesite.train_y = homesite.train_y.flatten()
    homesite.validation_y = homesite.validation_y.flatten()
    
#    reduced_range = range(0,100)
#    homesite.train_x = homesite.train_x[reduced_range]
#    homesite.train_y = homesite.train_y[reduced_range]

    C = [0.2,0.4,0.6,0.8,1] 
    for c in C:  
        # Creating classifier.
        clf = svm.SVC(kernel='linear',class_weight='balanced',C=c )    

        # Train classifier.
        print "Training classifier."
        clf.fit(homesite.train_x, homesite.train_y)

        # Test classifier.
        print 'Testing classifier.'
        predicted_labels = clf.predict(homesite.validation_x)

        # Show final results.
        results = confusion_matrix(homesite.validation_y, np.round(predicted_labels))
        accuracy, precision, recall = compute_performance_metrics(results)
        auc = compute_auc(homesite.validation_y, predicted_labels)
    
        result = [c,precision,recall,accuracy,auc]
        wr.writerow(result)        
        
        save_np_array("results/svm_accuracy.bin", np.array(accuracy_history))
        save_np_array("results/svm_precision.bin", np.array(precision_history))
        save_np_array("results/svm_recall.bin", np.array(recall_history))
        save_np_array("results/svm_auc.bin", np.array(auc_history))

        del clf
