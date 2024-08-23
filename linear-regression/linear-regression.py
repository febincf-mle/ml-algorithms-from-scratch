import numpy as np


class LinearRegression:


    def __init__(self, method='LSM', max_iteration=1000):
        """Constructor for LinearRegression which taked method as an argument
        method= LSM | GD | SGD, if chosen GD or SGD should specify max_iteration
        default max_iteration=1000"""

        self.theta = None
        self.method = method

        self.max_iteration = max_iteration if method == 'GD' else max_iteration * 10


    def least_squares(self, X, y):
        """Perform the least squares approximation for the given data and labels
        X must be n x d matrix where n -> n0. of data points, d -> features"""

        inv = np.linalg.pinv(X.T @ X)

        return inv @ X.T @ y
    
    
    def fit(self, data, labels):
        """Fits the data and computes the theta, To predict -> invoke self.predict method"""

        data = np.column_stack((np.ones(data.shape[0]), data))

        if self.method == 'LSM':
            self.theta = self.least_squares(data, labels)
        
        if self.method == 'GD':
            self.theta = self.gradient_descent(data, labels)
        
        if self.method == 'SGD':
            self.theta = self.stochastic_gradient_descent(data, labels)

        return True
    

    def predict(self, test):
        """Predicts the value for the test point based on the trained data"""

        test = np.column_stack(([1], test)).ravel()
        
        return test @ self.theta