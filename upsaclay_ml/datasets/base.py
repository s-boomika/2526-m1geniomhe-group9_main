"""
Dataset classes and utilities for loading and handling data .
this module provides base dataset classes and implementations for common datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import gzip
import pickle


class Dataset:
    """
    Base class for all datasets.
    
    Parameters
    ----------
    data : array-like or str
        Either the data itself or path/URL to a CSV file.
    target : array-like or str, optional
        Target values or name of target column in CSV.
        
    Attributes
    ----------
    X : ndarray
        Feature data.
    y : ndarray
        Target data (if available).
    """
    
    def __init__(self, data=None, target=None):
        """Initialize dataset."""
        self.X = None
        self.y = None
        
        if data is not None:
            if isinstance(data, str):
                # Load from CSV file or URL
                self._load_from_csv(data, target)
            else:
                # Direct data
                self.X = np.asarray(data)
                if target is not None:
                    self.y = np.asarray(target)
    
    def _load_from_csv(self, path_or_url, target_col=None):
        """
        Load data from CSV file or URL.
        
        Parameters
        ----------
        path_or_url : str
            Path to local CSV file or URL.
        target_col : str, optional
            Name of the target column.
        """
        # Check if it's a URL
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            df = pd.read_csv(path_or_url)
        else:
            df = pd.read_csv(path_or_url)
        
        if target_col and target_col in df.columns:
            self.y = df[target_col].values
            self.X = df.drop(columns=[target_col]).values
        else:
            self.X = df.values
            self.y = None
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X) if self.X is not None else 0
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample.
            
        Returns
        -------
        sample : tuple or ndarray
            (X, y) if y is available, otherwise just X.
        """
        if self.X is None:
            raise IndexError("Dataset is empty.")
        
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
    
    def __repr__(self):
        """Return string representation of the dataset."""
        n_samples = len(self)
        n_features = self.X.shape[1] if self.X is not None and self.X.ndim > 1 else 0
        has_target = self.y is not None
        
        return (f"{self.__class__.__name__}(n_samples={n_samples}, "
                f"n_features={n_features}, has_target={has_target})")
    
    def __str__(self):
        """Return detailed string representation with statistics."""
        if self.X is None:
            return f"{self.__class__.__name__}(empty)"
        
        n_samples = len(self)
        n_features = self.X.shape[1] if self.X.ndim > 1 else 1
        
        info = [
            f"{self.__class__.__name__} Statistics:",
            f"  Number of samples: {n_samples}",
            f"  Number of features: {n_features}",
        ]
        
        if self.y is not None:
            unique_targets = np.unique(self.y)
            info.append(f"  Number of classes/unique targets: {len(unique_targets)}")
            info.append(f"  Class distribution:")
            for target in unique_targets:
                count = np.sum(self.y == target)
                info.append(f"    Class {target}: {count} samples ({100*count/n_samples:.1f}%)")
        
        return "\n".join(info)


class DataLoader:
    """
    Data loader for batching and iterating over datasets.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to load from.
    batch_size : int, default=32
        How many samples per batch to load.
    shuffle : bool, default=False
        Set to True to have the data reshuffled at every epoch.
    transform : callable, optional
        Optional transform to be applied on a sample.
        
    Attributes
    ----------
    dataset : Dataset
        The input dataset.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle data.
    transform : callable
        Transform function.
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=False, transform=None):
        """Initialize DataLoader with given parameters."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        
        self.indices = np.arange(len(dataset))
    
    def __len__(self):
        """Return the number of batches."""
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __iter__(self):
        """
        Iterate over batches.
        
        Yields
        ------
        batch : ndarray or tuple of ndarrays
            A batch of data. Format depends on whether dataset has targets.
        """
        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Iterate over batches
        for start_idx in range(0, len(self.dataset), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Gather batch data
            if self.dataset.y is not None:
                batch_X = self.dataset.X[batch_indices]
                batch_y = self.dataset.y[batch_indices]
                
                if self.transform:
                    batch_X = self.transform(batch_X)
                
                yield batch_X, batch_y
            else:
                batch_X = self.dataset.X[batch_indices]
                
                if self.transform:
                    batch_X = self.transform(batch_X)
                
                yield batch_X
    
    def __repr__(self):
        """Return string representation."""
        return (f"DataLoader(dataset={self.dataset.__class__.__name__}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")
