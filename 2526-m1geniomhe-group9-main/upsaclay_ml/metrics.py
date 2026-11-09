"""
Metrics for evaluating model performance.

This module provides various metrics for supervised and unsupervised learning.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy classification score.
    
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the
    corresponding set of labels in y_true.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    score : float
        Accuracy score (fraction of correctly classified samples).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, "
                        f"y_pred has shape {y_pred.shape}")
    
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error regression loss.
    
    MSE = 1/n * Σ(y_true - y_pred)²
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    loss : float
        Mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, "
                        f"y_pred has shape {y_pred.shape}")
    
    return np.mean((y_true - y_pred) ** 2)


def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Coefficient of all samples.
    
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature array.
    labels : array-like of shape (n_samples,)
        Predicted cluster labels for each sample.
        
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
        Range: [-1, 1]
        - 1: clusters are well separated
        - 0: clusters are overlapping
        - -1: samples may be assigned to wrong clusters
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters >= n_samples:
        raise ValueError("Number of clusters must be between 2 and n_samples-1")
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Current sample and its cluster
        sample = X[i]
        cluster = labels[i]
        
        # (a) Mean distance to samples in same cluster
        same_cluster_mask = (labels == cluster)
        same_cluster_samples = X[same_cluster_mask]
        
        if same_cluster_samples.shape[0] > 1:
            # Exclude the sample itself
            distances = np.linalg.norm(same_cluster_samples - sample, axis=1)
            a = np.sum(distances) / (same_cluster_samples.shape[0] - 1)
        else:
            a = 0
        
        # (b) Mean distance to samples in nearest cluster
        b = np.inf
        for other_cluster in unique_labels:
            if other_cluster == cluster:
                continue
            
            other_cluster_mask = (labels == other_cluster)
            other_cluster_samples = X[other_cluster_mask]
            
            distances = np.linalg.norm(other_cluster_samples - sample, axis=1)
            mean_dist = np.mean(distances)
            
            if mean_dist < b:
                b = mean_dist
        
        # Silhouette coefficient for this sample
        if max(a, b) > 0:
            silhouette_vals[i] = (b - a) / max(a, b)
        else:
            silhouette_vals[i] = 0
    
    return np.mean(silhouette_vals)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error regression loss.
    
    MAE = 1/n * Σ|y_true - y_pred|
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    loss : float
        Mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, "
                        f"y_pred has shape {y_pred.shape}")
    
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) regression score.
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res is the residual sum of squares and
    SS_tot is the total sum of squares.
    
    Best possible score is 1.0. Can be negative (model performs worse than
    predicting the mean).
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    score : float
        R² score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, "
                        f"y_pred has shape {y_pred.shape}")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)
