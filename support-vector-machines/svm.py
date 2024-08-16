import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, minimize 


# Implementation of svm hard clustering model

class SVM:

    def __init__(self):
        """ Constructor for the SVM class which takes 0 arguments"""
        self.w = None
        self.alpha = None
        self.Q = None


    def compute_matrix_Y(self, labels):
        """ Computes the matrix Y, a diagonal matrix whose entries are from the given labels """
        n = labels.shape[0]
        Y = np.eye(n)

        for i in range(n):
            Y[i, i] = labels[i]

        return Y

    
    def compute_matrix_Q(self, data, labels):
        """Compute the matrix Q which is of the form Y.T @ X @ X.T @ Y"""
        Y = self.compute_matrix_Y(labels)
        self.Q = Y.T @ data.T @ data @ Y

        return 0
    

    def loss(self, alpha):
        """Computes the loss function for alpha which we want to optmize"""
        return (0.5 * alpha.T @ self.Q @ alpha) - alpha.T @ np.ones(alpha.shape[0])
    

    def jac(self, alpha):
        """Computes the gradient for the alpha function which we want to optimize"""
        return self.Q @ alpha - np.ones(alpha.shape[0])
    

    def optimize(self, n):
        """Optimize the objective function of alpha w.r.t contraints"""
        return minimize(self.loss, jac=self.jac, x0=np.zeros(n), method='SLSQP', bounds=Bounds(0, np.inf)).x
    

    def compute_weight_vector(self, data, Y):
        """ Computes the weight vector w which we use to predict the class"""
        return data @ Y @ self.alpha
    

    def fit(self, data, labels):
        """Fits the data and ready for prediction to predict use predict method"""

        n = data.shape[0]
        data = data.T # Easy to work with d x n than n x d

        Y = self.compute_matrix_Y(labels)
        self.compute_matrix_Q(data, labels)

        self.alpha = self.optimize(n)
        self.w = self.compute_weight_vector(data, Y)

        return 0
    

    def predict(self, test_point):
        """Predicts the class of the datapoint either -1 or 1"""
        return np.sign(self.w @ test_point)