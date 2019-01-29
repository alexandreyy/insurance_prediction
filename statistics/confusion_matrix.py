'''
Created on 16/11/2015

@author: Alexandre Yukio Yamashita
'''

from sklearn.metrics import confusion_matrix as c_matrix


def confusion_matrix(true_labels, predicted_labels):
    '''
    Print and get confusion matrix.
    '''
    result = c_matrix(true_labels, predicted_labels)

    # Show final results.
    print "\nConfusion matrix:"
    print "(real , predicted) | count"
    print "(%s , %s) | %f" % ("False", "False", result[0, 0] * 1.0 / (result[0, 0] + result[0, 1]))
    print "(%s , %s) | %f" % ("False", "True", result[0, 1] * 1.0 / (result[0, 0] + result[0, 1]))
    print "(%s , %s) | %f" % ("True", "False", result[1, 0] * 1.0 / (result[1, 0] + result[1, 1]))
    print "(%s , %s) | %f" % ("True", "True", result[1, 1] * 1.0 / (result[1, 0] + result[1, 1]))

    return result