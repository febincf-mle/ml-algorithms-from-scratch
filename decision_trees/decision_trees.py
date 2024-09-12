import numpy as np
from collections import Counter
from decision_trees import DecisionTreeClassifier



class Node:

    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        constructor for node class for Decision Trees"""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def _is_leaf(self):
       """
       Returns True if a node is leaf or False otherwise"""
       if self.value == None:
           return False
       return True
    


class DecisionTreeClassifier:


    def __init__(self, max_depth=100, min_sample_split=2, n_features=None):
        """
        Constructor for DecisionTreeClassifier, takes 3 arguments \n
        parameters\n
        - max_depth -> maximum depth of the decision tree
        - min_sample_split -> specifies when to stop splitting a node"""

        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.root = None


    def _best_split(self, X, y, feat_idxs):
        """
        Returns the best feature to split and the threshold value to split the node.\n
        The best split is calculated using Information gain."""

        max_gain = -1
        best_feature, best_threshold = None, None

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            X_column = X[:, feat_idx]

            for threshold in thresholds:

                information_gain = self._information_gain(X_column, y, threshold)

                if information_gain > max_gain:
                    max_gain = information_gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_threshold, best_feature


    def _information_gain(self, X, y, threshold):
        """
        Returns the information gain of a particular split.\n
        Calculated based on entropy"""

        n = len(y)

        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        parent_entropy = self._calculate_entropy(y)
        e_l, e_r = self._calculate_entropy(y[left_idxs]), self._calculate_entropy(y[right_idxs])
        nl, nr = len(left_idxs), len(right_idxs)

        child_entropy = (nl / n) * e_l + (nr / n) * e_r

        return parent_entropy - child_entropy


    def _calculate_entropy(self, y):
        """
        Calculates the entropy of a node"""

        proportions = np.bincount(y) / len(y)
        return -np.sum([np.log(p) * p for p in proportions if p > 0])


    def _max_count_value(self, y):
        """
        Returns the most common label in a leaf node."""

        counter = Counter(y)
        return counter.most_common(1)[0][0]
    

    def _grow_tree(self, X, y, depth=0):
        """
        Grows the decision tree up to specified max_depth parameter"""

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_samples < self.min_sample_split or n_labels == 1):
            value = self._max_count_value(y)
            return Node(value=value)
        
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        best_threshold, best_feature = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = np.argwhere(X[:, best_feature] <= best_threshold).flatten(), np.argwhere(X[:, best_feature] > best_threshold).flatten()

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)
    

    def fit(self, X, y):
        """
        fits the data, creates the tree and ready for predictions\n
        parameters\n
        - X -> data matrix of shape n x d ( n -> no. of samples, d -> no. of features)
        - y -> labels for the data points"""
        
        n_features = X.shape[1]
        self.n_features = min(self.n_features, n_features) if self.n_features else n_features
        self.root = self._grow_tree(X, y)


    def predict(self, X):
        """
        predicts the samples based on the trained model"""
        predictions =  [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    

    def _traverse_tree(self, x, node):
        """
        Traverse through the tree and makes a decision. returns the label for the sample"""
        if node._is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)