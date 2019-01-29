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


'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''
from data.plot import plot
import csv
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from data.data import Data
from data.numpy_file import save_np_array,load_np_array
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_performance_metrics, compute_auc
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier
from unbalanced_dataset.over_sampling import OverSampler, SMOTE
from sklearn.externals import joblib
from unbalanced_dataset.under_sampling import TomekLinks


def plot_roc(mean_fpr,mean_tpr,mean_auc):
      plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic example')
      plt.legend(loc="lower right")
      plt.show() 
      
      
def reportBestResult():
     bestAUC = 0
     bestParam = 0
     C = [0.2,0.4,0.6,0.8,1]
     for c in C: 
         auc_history = load_np_array("results/svm_auc_"+str(c)+".bin")
         mean_auc = auc_history.mean()
         if(mean_auc > bestAUC):
             bestAUC = mean_auc
             bestParam = c
    
     print bestParam
     
     confusion_matrix_history = load_np_array("results/svm_folds_confusion_"+str(c)+".bin")
     print confusion_matrix_history
     
     mean_cm = np.mean(confusion_matrix_history, axis = 2)
     std_cm = np.std(confusion_matrix_history, axis = 2)
#     for i in range(0,2):
#         for j in range(0,2):
#            mean_cm[i][j] = confusion_matrix_history[i][j].mean()
            
     print mean_cm
     print std_cm
     compute_performance_metrics(mean_cm)
     acc_mean = []
     recall_mean = []
     auc_mean = []
     for c in C: 
         accuracy_history = load_np_array("results/svm_accuracy_"+str(c)+".bin")
         recall_history = load_np_array("results/svm_recall_"+str(c)+".bin")
         auc_history = load_np_array("results/svm_auc_"+str(c)+".bin")
         acc_mean.append(accuracy_history.mean())
         recall_mean.append(recall_history.mean())
         auc_mean.append(auc_history.mean())
         
     
     print auc_mean
#     plot("results/random_forest_balacing.png", [acc_mean, recall_mean], \
#         ["sensitividade ", "AUC"], "taxa de balanceamento", "metricas", 'center right')
     
      
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
    confusion_matrix_history = np.array([])
    confusion_matrix_history.shape = (2, 2, 0)
    
    path = "resources/parsed_data.bin"
    homesite = Data()
    homesite.load_parsed_data(path)
    homesite.z_norm_train_test_by_feature()
    del homesite.test_x 
    # Deleted to save memory.

#    reduced_range = range(0,100)
#    homesite.train_x = homesite.train_x[reduced_range]
#    homesite.train_y = homesite.train_y[reduced_range]    
    C = [0.2,0.4,0.6,0.8,1]
    for c in C:  
        # Creating classifier.
        mean_acc = 0.0
        mean_recall = 0.0
        mean_precision = 0.0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        
        cvs = StratifiedKFold(homesite.train_y,n_folds=5)
        
        clf = svm.SVC(kernel='linear',probability=True,C=c )
        
        # Train classifier.
        print "\nTraining classifier param %d" % c
        for i,(train, test) in enumerate(cvs):
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
            
            save_np_array("results/svm_accuracy_"+str(c)+".bin", np.array(accuracy_history))
            save_np_array("results/svm_precision_"+str(c)+".bin", np.array(precision_history))
            save_np_array("results/svm_recall_"+str(c)+".bin", np.array(recall_history))
            save_np_array("results/svm_auc_"+str(c)+".bin", np.array(auc_history))
            
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            
        mean_acc /= len(cvs)    
        mean_recall /= len(cvs)    
        mean_precision /= len(cvs)    
        mean_tpr /= len(cvs)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
#        
        print 'AUC MEDIO'
#        print mean_acc
#        print mean_recall
#        print mean_precision
        print mean_auc
        
        plot_roc(mean_fpr,mean_tpr,mean_auc)
        
        save_np_array("results/svm_folds_confusion_"+str(c)+".bin", np.array(confusion_matrix_history))
        filename = 'D:\MACHINE_LEARNING\ML_FINAL_PROJECT\classifiers\svm_'+str(c)+'.joblib.pkl'
        joblib.dump(clf, filename, compress=9)
        del clf
        
    
    
#    confusion_matrix_history = load_np_array("results/rfc_folds_confusion_128.bin")   
#    accuracy_history = load_np_array("results/svm_accuracy_128.bin")
#    precision_history = load_np_array("results/svm_precision_128.bin")
#    recall_history = load_np_array("results/svm_recall_128.bin")
#    auc_history = load_np_array("results/svm_auc_128.bin")
#    
#    
#    plot("results/random_forest_balacing.png", [recall_history, auc_history], \
#         ["sensitividade ", "AUC"], "taxa de balanceamento", "metricas", 'center right')
    
    reportBestResult()    
    
    results_f.close()

