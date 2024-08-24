import numpy as np

def accuracy(y_test, y_pred):
    """
    Return the percentage of accuracy of a classification model\n
    - y_test -> test labels
    - y_pred -> predicted labels of test data"""
    return np.sum(y_test == y_pred) / len(y_pred)