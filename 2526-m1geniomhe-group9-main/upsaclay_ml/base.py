"""
Base classes for estimators and transformers in the upsaclay_ml library.

This module defines the abstract base classes that all models and transformers
must inherit from, ensuring a consistent interface across the library.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEstimator(ABC):
    """
    Abstract base class for all estimators (models) in upsaclay_ml.
    
    All estimators should inherit from this class and implement fit() and predict().
    This ensures a consistent interface similar to scikit-learn.
    """
    
    def __init__(self):
        """Initialize the base estimator."""
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the estimator to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (for supervised learning).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        pass
    
    def fit_predict(self, X, y=None):
        """
        Fit the estimator and return predictions on the same data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        self.fit(X, y)
        return self.predict(X)
    
    @property
    def is_fitted(self):
        """Check if the estimator has been fitted."""
        return self._is_fitted
    
    def __repr__(self):
        """Return string representation of the estimator."""
        params = self._get_param_names()
        param_str = ", ".join(f"{k}={getattr(self, k, None)}" for k in params)
        return f"{self.__class__.__name__}({param_str})"
    
    def __str__(self):
        """Return human-readable string representation."""
        return self.__repr__()
    
    def __eq__(self, other):
        """
        Compare estimators by class and hyperparameters.
        
        Parameters
        ----------
        other : object
            Another estimator to compare with.
            
        Returns
        -------
        equal : bool
            True if estimators have same class and hyperparameters.
        """
        if not isinstance(other, self.__class__):
            return False
        
        params = self._get_param_names()
        return all(getattr(self, k, None) == getattr(other, k, None) for k in params)
    
    def __call__(self, X):
        """Shortcut for predict() method."""
        return self.predict(X)
    
    def _get_param_names(self):
        """Get parameter names for __repr__ and __eq__."""
        # Get init signature parameters (excluding 'self')
        import inspect
        sig = inspect.signature(self.__init__)
        return [p.name for p in sig.parameters.values() if p.name != 'self']


class BaseTransformer(ABC):
    """
    Abstract base class for all transformers in upsaclay_ml.
    
    All transformers should inherit from this class and implement fit() and transform().
    """
    
    def __init__(self):
        """Initialize the base transformer."""
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (ignored, present for API consistency).
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_out)
            Transformed data.
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_out)
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def is_fitted(self):
        """Check if the transformer has been fitted."""
        return self._is_fitted
    
    def __repr__(self):
        """Return string representation of the transformer."""
        params = self._get_param_names()
        param_str = ", ".join(f"{k}={getattr(self, k, None)}" for k in params)
        return f"{self.__class__.__name__}({param_str})"
    
    def __str__(self):
        """Return human-readable string representation."""
        return self.__repr__()
    
    def __eq__(self, other):
        """Compare transformers by class and parameters."""
        if not isinstance(other, self.__class__):
            return False
        
        params = self._get_param_names()
        return all(getattr(self, k, None) == getattr(other, k, None) for k in params)
    
    def _get_param_names(self):
        """Get parameter names for __repr__ and __eq__."""
        import inspect
        sig = inspect.signature(self.__init__)
        return [p.name for p in sig.parameters.values() if p.name != 'self']
