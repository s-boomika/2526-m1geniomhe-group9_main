"""
Decision Tree classifier for supervised learning.
this module implements a Decision Tree classifier using recursive partitioning
based on information gain or Gini impurity.
"""

import numpy as np
from .supervised import SupervisedModel


class Node:
    """
    A node in the decision tree.
    
    Parameters
    ----------
    feature : int, optional
        Index of the feature to split on.
    threshold : float, optional
        Threshold value for the split.
    left : Node, optional
        Left child node.
    right : Node, optional
        Right child node.
    value : int or float, optional
        Class label (for leaf nodes).
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """Initialize a tree node."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier(SupervisedModel):
    """
    A decision tree classifier.
    
    The decision tree recursively splits the data based on feature values
    to create a tree structure for classification.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
        
    Attributes
    ----------
    tree_ : Node
        The root node of the decision tree.
    n_classes_ : int
        The number of classes.
    feature_importances_ : ndarray of shape (n_features,)
        Normalized total reduction of criterion by feature.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        """Initialize Decision Tree Classifier with given parameters."""
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        
        if criterion not in ['gini', 'entropy']:
            raise ValueError(f"Criterion must be 'gini' or 'entropy', got {criterion}")
        
        self.tree_ = None
        self.n_classes_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_classes_ = len(np.unique(y))
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        
        # Build the tree recursively
        self.tree_ = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        depth : int
            Current depth in the tree.
            
        Returns
        -------
        node : Node
            The root node of the (sub)tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            # Create leaf node with most common class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            # No valid split found, create leaf
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        best_feature : int or None
            Index of the best feature to split on.
        best_threshold : float or None
            Best threshold value for the split.
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Calculate current impurity
        current_impurity = self._calculate_impurity(y)
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Check all features
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try all unique values as thresholds
            for threshold in thresholds:
                # Split the data
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate information gain
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                impurity_left = self._calculate_impurity(y[left_indices])
                impurity_right = self._calculate_impurity(y[right_indices])
                
                weighted_impurity = (n_left / n_samples) * impurity_left + \
                                   (n_right / n_samples) * impurity_right
                
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Update feature importances
        if best_feature is not None:
            self.feature_importances_[best_feature] += best_gain
        
        return best_feature, best_threshold
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity of a set of labels.
        
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        impurity : float
            Gini impurity or entropy.
        """
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y) / len(y)
        
        if self.criterion == 'gini':
            # Gini impurity: 1 - sum(p_i^2)
            return 1 - np.sum(proportions ** 2)
        elif self.criterion == 'entropy':
            # Entropy: -sum(p_i * log2(p_i))
            # Avoid log(0)
            proportions = proportions[proportions > 0]
            return -np.sum(proportions * np.log2(proportions))
    
    def _most_common_label(self, y):
        """Return the most common label in y."""
        return np.bincount(y).argmax()
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.
        
        Parameters
        ----------
        x : ndarray of shape (n_features,)
            A single sample.
        node : Node
            Current node in the tree.
            
        Returns
        -------
        prediction : int
            Predicted class label.
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
