"""
Linear models for regression and classification.
this module implements linear models including Linear Regression.
"""

import numpy as np
from .supervised import SupervisedModel


class LinearRegression(SupervisedModel):
    """
    Ordinary least squares Linear Regression.
    
    LinearRegression fits a linear model with coefficients to minimize
    the residual sum of squares between the observed targets and the
    targets predicted by the linear approximation.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    """
    
    def __init__(self, fit_intercept=True):
        """Initialize Linear Regression with given parameters."""
        super().__init__()
        self.fit_intercept = fit_intercept
        
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit linear model using ordinary least squares.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        # Solve using normal equation: theta = (X^T X)^(-1) X^T y
        try:
            theta = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = theta
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
