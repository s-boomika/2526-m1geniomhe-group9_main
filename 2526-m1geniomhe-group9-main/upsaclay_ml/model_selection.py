"""
Model selection utilities for cross-validation and data splitting.

This module provides tools for splitting datasets and validating models.
"""

import numpy as np


def train_test_split(*arrays, test_size=0.25, shuffle=True, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices.
    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Controls the shuffling applied to the data before splitting.
        Pass an int for reproducible output.
        
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        For example, for X, y inputs: [X_train, X_test, y_train, y_test]
        
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    
    # Convert to numpy arrays
    arrays = [np.asarray(arr) for arr in arrays]
    
    # Check that all arrays have the same length
    n_samples = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError(f"All arrays must have the same length. "
                           f"Got different lengths: {[arr.shape[0] for arr in arrays]}")
    
    # Determine test size
    if isinstance(test_size, float):
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if test_size < 0 or test_size > n_samples:
            raise ValueError(f"test_size must be between 0 and {n_samples}, got {test_size}")
        n_test = test_size
    else:
        raise TypeError(f"test_size must be float or int, got {type(test_size)}")
    
    n_train = n_samples - n_test
    
    # Create indices
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split arrays
    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    
    return result


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    """
    Evaluate a score by cross-validation.
    
    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data. Must implement fit and predict.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,)
        The target variable.
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : callable, optional
        A scorer callable object with signature scorer(y_true, y_pred).
        If None, uses accuracy for classification.
        
    Returns
    -------
    scores : ndarray of shape (cv,)
        Array of scores for each fold.
    """
    from .metrics import accuracy_score
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    if scoring is None:
        scoring = accuracy_score
    
    n_samples = X.shape[0]
    fold_size = n_samples // cv
    
    scores = []
    
    for i in range(cv):
        # Create train/test split for this fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < cv - 1 else n_samples
        
        test_indices = np.arange(test_start, test_end)
        train_indices = np.concatenate([np.arange(0, test_start), 
                                       np.arange(test_end, n_samples)])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Fit and predict
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        # Score
        score = scoring(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)
