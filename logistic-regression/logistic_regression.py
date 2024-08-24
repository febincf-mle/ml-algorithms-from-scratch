import numpy as np


class LogisticRegression:

    def __init__(self, rate=0.05, max_iterations=1000):
        """
        Constructor method for the LogisticRegression class\n
        parameters\n
        -rate -> learning rate alpha (0.05)
        -max_iterations -> maximum iterations for the GD to converge"""

        self.rate = rate
        self.max_iterations = max_iterations
        self.weights = None


    def compute_sigmoid(self, x):
        """
        Computes the sigmoid value for the datapoints"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))


    def fit(self, X, y):
        """
        Fits the data and updates the weight vector\n
        parameters\n
        - X -> data matrix of order n x d
            - n -> no. of data points
            - d -> no. of features
            
        - y -> labels"""

        n, d = X.shape

        self.weights = np.zeros(d)

        for _ in range(self.max_iterations):

            linear_predictions = np.dot(X, self.weights)
            predictions = self.compute_sigmoid(linear_predictions)

            dw = (1 / n) * np.dot(X.T, predictions - y) 

            self.weights = self.weights - dw * self.rate


    def predict(self, X):
        """
        Returns the predicted labels of the data matrix provided based on Trained model"""

        linear_predictions = np.dot(X, self.weights)
        sigmoid_predictions = self.compute_sigmoid(linear_predictions)

        labels = [1 if i >0.5 else 0 for i in sigmoid_predictions]

        return labels