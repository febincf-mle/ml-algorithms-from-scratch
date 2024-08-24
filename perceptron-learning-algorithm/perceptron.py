import numpy as np


class Perceptron:

    
    def __init__(self, epochs=1000):
        """
        Constructor method for Perceptron class
        epochs -> int, specifies the max number of iterations"""
       
        self.epochs = epochs
        self.theta = None
        self.updates_per_epoch = None


    def init_theta(self, d):
        """
        Initializes the theta to be a zero vector of d dimensions"""
        return np.zeros((d,))
    

    def compute_sign(self, x):
        """
        Computes the label of the value based on the sign"""

        if x >= 0:
            return 1
        
        return 0
    

    def perform_perceptron(self, data, labels, n, d):
        """
        Performs the perceptron learning algorithm and returns the theta vector
        and updates in each iteration (array)"""

        theta, updates_per_epoch = self.init_theta(d), []

        for _ in range(self.epochs):

            updates_in_ith_epoch = 0

            for i in range(n):

                predicted_y = self.compute_sign(theta @ data[i])

                if predicted_y != labels[i]:

                    theta += (labels[i] - predicted_y) * data[i]
                    updates_in_ith_epoch += 1

            updates_per_epoch.append(updates_in_ith_epoch)

            if updates_in_ith_epoch == 0:
                return updates_per_epoch, theta


    def fit(self, data, labels):
        """
        Fits the data performs the algorithm \n
        parameters 
            - data ->  A Matrix of shape n x d where n - no of data points, d - features
            - labels -> Corresponding labels of data Matrix"""

        n, d = data.shape
        self.updates_per_epoch, self.theta = self.perform_perceptron(data, labels, n, d)

        return True
    

    def predict(self, test):
        """
        Classifies the given test point based on the trained Model \n
        The vector test -> should be d - dimensional,
        where d is the number of features in training data"""
        
        return self.compute_sign(self.theta @ test)