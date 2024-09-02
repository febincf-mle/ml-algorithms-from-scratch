import numpy as np


class Kmeans:

    
    def __init__(self, k_clusters=2):
        """
        Constructor for the Kmeans class \n
        takes in one parameter\n
        k_clusters: int -> specifies the number of clusters you want"""

        self.k_clusters = k_clusters
        self.centroids = None
        self.labels = None


    def initialize_centroids(self, X, n):
        """
        Initialize k random centroids from the given data"""

        idx = np.arange(0, n)
        np.random.shuffle(idx)

        return X[idx[: self.k_clusters]]
    

    def compute_distance_from_mean(self, xi, centroids):
        """
        Computes the eucledian distance of the given point xi from the k different cluster centers\n
        parameters:\n
        - xi -> the datapoint
        - centroids -> cluster centers"""

        return np.linalg.norm(xi - centroids, axis=1)
    

    def has_convergence_achieved(self, diff_in_centroids):
        """
        Returns true if the convergence criteria has achieved"""
        return np.sum(np.linalg.norm(diff_in_centroids, axis=1)) == 0


    def fit(self, X):
        """
        Fits the given data and performs the Kmeans clustering algorithm\n
        parameters: \n
        - X -> data matrix of order n x d (n = no. of datapoints, d = no. of features)"""

        n, d = X.shape
        k = self.k_clusters

        centroids = self.initialize_centroids(X, n)
        self.labels = np.zeros(n, dtype=int)


        while True:

            for i in range(n):

                distance_from_mean = self.compute_distance_from_mean(X[i], centroids)
                self.labels[i] = np.argmin(distance_from_mean)


            updated_centroids = []
            for k in range(self.k_clusters):

                mu_k = X[self.labels == k]
                updated_centroids.append(mu_k.mean(axis=0))
                

            updated_centroids = np.array(updated_centroids)
            if self.has_convergence_achieved(updated_centroids - centroids):
                self.centroids = updated_centroids
                break

            centroids = updated_centroids

        return True