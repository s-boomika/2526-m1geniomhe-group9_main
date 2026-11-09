"""
upsaclay_ml: A Machine Learning Library Built with OOP Principles

This library provides implementations of common machine learning algorithms,
transformers, and utilities following object-oriented programming best practices.
"""

__version__ = "1.0.0"

from .base import BaseEstimator, BaseTransformer

__all__ = [
    "BaseEstimator",
    "BaseTransformer",
]
