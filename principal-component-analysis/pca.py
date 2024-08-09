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


    def compute_covariance_matrix(self, X):
        """ Return the covariance matrix of the dataset """
        return ( X.T @ X ) * (1 / X.shape[0]) 
    

    def eigen_decomposition(self, C):
        """Returns the eigenvalues and eigenvectors of the given Covariance-Matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][: self.k_components]
        eigenvectors = eigenvectors[:, idx][:, :self.k_components]

        return eigenvalues, eigenvectors


    def fit(self, X):
        """Fits the dataset and compute the eigenvalues and eigenvectors"""
        self._centered_dataset = self.center_dataset(X)
        covariance_matrix = self.compute_covariance_matrix(self._centered_dataset)

        self._eigenvalues, self._eigenvectors = self.eigen_decomposition(covariance_matrix)
        return 0


    def transform(self, centered_dataset, eigenvectors):
        """Transforms the dataset by reducing the n dimenstion to specified k dimension"""
        return centered_dataset @ eigenvectors