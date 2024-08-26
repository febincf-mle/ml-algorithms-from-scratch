# Importing required packages
import numpy as np
import matplotlib.pyplot as plt


# Implementation of pca


class PCA:

    def __init__(self, k_components=2):
        """ Constructor for PCA which takes in only one paramter k_components which is the number
        of components you want to preserve."""
        self.k_components = k_components
        self._eigenvalues = None
        self._eigenvectors = None
        self._centered_dataset = None


    def center_dataset(self, X):
        """ perform a centering of dataset by subtracting each data point from its mean value"""
        return X - X.mean(axis=0)
    
    
    def compute_covariance_matrix(self, X, alt=False):
        """
        Computes the matrix X @ X.T in case the d >> n\n
        - Each of the eigenvalues of covariance will be scaled by n, because C = n * X @ X.T,
        both XX.T and X.TX have the same eigenvalues"""
        if alt: return np.dot(X, X.T)
        return (1 / X.shape[0]) *  np.dot(X.T, X)
    

    def eigen_decomposition(self, C, alt=False):
        """Returns the eigenvalues and eigenvectors of the given Covariance-Matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][: self.k_components]
        eigenvectors = eigenvectors[:, idx][:, :self.k_components]

        if  alt:
            normalized_eigenvectors = eigenvectors / np.sqrt(eigenvalues)
            eigenvectors = np.dot(self._centered_dataset.T, normalized_eigenvectors)

        return eigenvalues, eigenvectors


    def fit(self, X):
        """Fits the dataset and compute the eigenvalues and eigenvectors"""

        n, d = X.shape

        alt = (d / n) > 100

        self._centered_dataset = self.center_dataset(X)
        
        C = self.compute_covariance_matrix(self._centered_dataset, alt)
        self._eigenvalues, self._eigenvectors = self.eigen_decomposition(C, alt)


    def transform(self):
        """Transforms the dataset by reducing the n dimenstion to specified k dimension"""
        return self._centered_dataset @ self._eigenvectors