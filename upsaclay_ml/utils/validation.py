"""
validation utilities for the upsaclay_ml library in oder too be sure this can be consider as a sub first test for our library
this module provides helper functions to check input data and estimator states,
ensuring consistency and better error messages across the library.
"""

import numpy as np

def check_array(X, dtype='numeric'):
    """
    Input validation for standard array-like structures.
    
    Parameters
    ----------
    X : array-like
        The input data to validate.
    dtype : str, optional
        Target type of the array.
    
    Returns
    -------
    X_converted : numpy.ndarray
        The validated and converted numpy array.
    """
    if X is None:
        raise ValueError("Input array X cannot be None")
    
    try:
        X = np.asarray(X)
    except Exception as e:
        raise ValueError(f"Could not convert input to numpy array: {e}")
    
    if X.size == 0:
        raise ValueError("Input array X cannot be empty")
        
    if X.ndim == 0:
        raise ValueError("Singleton array (0-D) is not allowed")
        
    return X

def check_X_y(X, y):
    """
    Input validation for standard X and y pairs.
    Checks that X and y have consistent lengths.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
        
    Returns
    -------
    X, y : tuple of numpy.ndarray
        The validated arrays.
    """
    X = check_array(X)
    
    if y is None:
        raise ValueError("y cannot be None")
        
    y = np.asarray(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: X={X.shape[0]}, y={y.shape[0]}")
        
    return X, y

def check_is_fitted(estimator, attributes=None):
    """
    Check if an estimator is fitted.
    
    Parameters
    ----------
    estimator : estimator instance
        Estimator instance to check.
    attributes : str or list of str, optional
        Attribute(s) to check for existence. If None, checks for standard
        attributes ending in an underscore (like 'coef_', 'cluster_centers_').
        
    Raises
    ------
    ValueError
        If the estimator is not fitted.
    """
    if hasattr(estimator, 'is_fitted') and not estimator.is_fitted:
        raise ValueError(
            f"This {estimator.__class__.__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )
        
    if attributes is not None:
        if isinstance(attributes, str):
            attributes = [attributes]
            
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise ValueError(
                    f"This {estimator.__class__.__name__} instance is not fitted yet. "
                    f"Attribute '{attr}' not found."
                )
