"""
Transformers for data preprocessing and feature engineering.
this module implements various data transformation techniques including
scaling, normalization, and encoding.
"""

import numpy as np
from ..base import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
        
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    """
    
    def __init__(self, with_mean=True, with_std=True):
        """Initialize StandardScaler with given parameters."""
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming.")
        
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_


class MinMaxScaler(BaseTransformer):
    """
    Transform features by scaling each feature to a given range.
    
    This estimator scales each feature individually to the range [0, 1]
    by default, or to [feature_min, feature_max] if specified.
    
    The transformation is given by:
        X_scaled = (X - X.min) / (X.max - X.min)
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
        
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the training data.
    max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the training data.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum in the training set.
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum in the training set.
    data_range_ : ndarray of shape (n_features,)
        Per feature range (max - min) in the training set.
    """
    
    def __init__(self, feature_range=(0, 1)):
        """Initialize MinMaxScaler with given parameters."""
        super().__init__()
        self.feature_range = feature_range
        
        self.min_ = feature_range[0]
        self.max_ = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Avoid division by zero
        self.data_range_ = np.where(self.data_range_ == 0, 1, self.data_range_)
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming.")
        
        X = np.asarray(X)
        
        # Scale to [0, 1]
        X_std = (X - self.data_min_) / self.data_range_
        
        # Scale to [min, max]
        X_scaled = X_std * (self.max_ - self.min_) + self.min_
        
        return X_scaled


class Binarizer(BaseTransformer):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold.
    
    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0.
    
    Parameters
    ----------
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0,
        above it by 1.
        
    Attributes
    ----------
    None - Binarizer is a stateless transformer
    """
    
    def __init__(self, threshold=0.0):
        """Initialize Binarizer with given threshold."""
        super().__init__()
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """
        Fit the binarizer (no-op, binarizer is stateless).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Binarizer is stateless, but we still mark it as fitted
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Binarize each element of X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to binarize.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Binarized data.
        """
        X = np.asarray(X)
        return (X > self.threshold).astype(int)


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array.
    
    The input should be integers representing categories.
    
    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore unknown categories during transform.
        
    Attributes
    ----------
    categories_ : list of ndarrays
        The categories of each feature determined during fitting.
    n_features_out_ : int
        Number of features in the transformed output.
    """
    
    def __init__(self, handle_unknown='error'):
        """Initialize OneHotEncoder with given parameters."""
        super().__init__()
        self.handle_unknown = handle_unknown
        
        self.categories_ = None
        self.n_features_out_ = None
    
    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Determine categories for each feature
        self.categories_ = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)
        
        # Calculate total number of output features
        self.n_features_out_ = sum(len(cats) for cats in self.categories_)
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform X using one-hot encoding.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out)
            Transformed input with one-hot encoding.
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming.")
        
        X = np.asarray(X)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self.n_features_out_))
        
        col_idx = 0
        for feature_idx in range(X.shape[1]):
            categories = self.categories_[feature_idx]
            
            for sample_idx in range(n_samples):
                value = X[sample_idx, feature_idx]
                
                # Find which category this value belongs to
                try:
                    cat_idx = np.where(categories == value)[0][0]
                    result[sample_idx, col_idx + cat_idx] = 1
                except IndexError:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category {value} in feature {feature_idx}")
                    # If 'ignore', leave all zeros for this sample/feature
            
            col_idx += len(categories)
        
        return result
