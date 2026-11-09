"""
Common datasets for machine learning.
this module provides easy access to popular datasets like MNIST, Iris, and Blobs (or the CardI-HACK but it's not really popular as the other).
"""

import numpy as np
from .base import Dataset
import urllib.request
import gzip
import pickle
from pathlib import Path


class MNIST(Dataset):
    """
    MNIST handwritten digits dataset.
    
    The MNIST database contains 60,000 training images and 10,000 test images
    of handwritten digits (0-9).
    
    Parameters
    ----------
    subset : {'train', 'test'}, default='train'
        Which subset of the data to load.
    data_home : str, optional
        Directory to cache the dataset.
        
    Attributes
    ----------
    X : ndarray of shape (n_samples, 784)
        Flattened images (28x28 pixels).
    y : ndarray of shape (n_samples,)
        Labels (0-9).
    images : ndarray of shape (n_samples, 28, 28)
        Original image shape.
    """
    
    def __init__(self, subset='train', data_home=None):
        """Initialize MNIST dataset."""
        super().__init__()
        
        self.subset = subset
        self.data_home = data_home or str(Path.home() / '.upsaclay_ml' / 'mnist')
        
        # Create data directory if needed
        Path(self.data_home).mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load MNIST data from file or download if necessary."""
        # URLs for MNIST data
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        
        if self.subset == 'train':
            images_file = 'train-images-idx3-ubyte.gz'
            labels_file = 'train-labels-idx1-ubyte.gz'
        else:
            images_file = 't10k-images-idx3-ubyte.gz'
            labels_file = 't10k-labels-idx1-ubyte.gz'
        
        images_path = Path(self.data_home) / images_file
        labels_path = Path(self.data_home) / labels_file
        
        # Download if not exists
        if not images_path.exists():
            print(f"Downloading {images_file}...")
            urllib.request.urlretrieve(base_url + images_file, images_path)
        
        if not labels_path.exists():
            print(f"Downloading {labels_file}...")
            urllib.request.urlretrieve(base_url + labels_file, labels_path)
        
        # Load images
        with gzip.open(images_path, 'rb') as f:
            # Skip header (16 bytes)
            f.read(16)
            # Read image data
            buf = f.read()
            images = np.frombuffer(buf, dtype=np.uint8)
            images = images.reshape(-1, 28, 28)
        
        # Load labels
        with gzip.open(labels_path, 'rb') as f:
            # Skip header (8 bytes)
            f.read(8)
            # Read label data
            buf = f.read()
            labels = np.frombuffer(buf, dtype=np.uint8)
        
        # Store data
        self.images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
        self.X = images.reshape(-1, 784).astype(np.float32) / 255.0  # Flatten
        self.y = labels.astype(np.int64)


class Iris(Dataset):
    """
    Iris flower dataset.
    
    The Iris dataset consists of 150 samples from three species of Iris flowers:
    - Setosa (0)
    - Versicolor (1)
    - Virginica (2)
    
    Each sample has 4 features:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
    
    Attributes
    ----------
    X : ndarray of shape (150, 4)
        Feature data.
    y : ndarray of shape (150,)
        Target labels (0, 1, or 2).
    feature_names : list of str
        Names of the features.
    target_names : list of str
        Names of the target classes.
    """
    
    def __init__(self):
        """Initialize Iris dataset."""
        super().__init__()
        
        self.feature_names = [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ]
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load Iris data (hardcoded for reliability)."""
        # Iris dataset (hardcoded to avoid external dependencies)
        # Format: sepal_length, sepal_width, petal_length, petal_width, class
        iris_data = np.array([
            [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
            [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0],
            [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0],
            [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
            [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0],
            [5.7, 4.4, 1.5, 0.4, 0], [5.4, 3.9, 1.3, 0.4, 0], [5.1, 3.5, 1.4, 0.3, 0],
            [5.7, 3.8, 1.7, 0.3, 0], [5.1, 3.8, 1.5, 0.3, 0], [5.4, 3.4, 1.7, 0.2, 0],
            [5.1, 3.7, 1.5, 0.4, 0], [4.6, 3.6, 1.0, 0.2, 0], [5.1, 3.3, 1.7, 0.5, 0],
            [4.8, 3.4, 1.9, 0.2, 0], [5.0, 3.0, 1.6, 0.2, 0], [5.0, 3.4, 1.6, 0.4, 0],
            [5.2, 3.5, 1.5, 0.2, 0], [5.2, 3.4, 1.4, 0.2, 0], [4.7, 3.2, 1.6, 0.2, 0],
            [4.8, 3.1, 1.6, 0.2, 0], [5.4, 3.4, 1.5, 0.4, 0], [5.2, 4.1, 1.5, 0.1, 0],
            [5.5, 4.2, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0],
            [5.5, 3.5, 1.3, 0.2, 0], [4.9, 3.6, 1.4, 0.1, 0], [4.4, 3.0, 1.3, 0.2, 0],
            [5.1, 3.4, 1.5, 0.2, 0], [5.0, 3.5, 1.3, 0.3, 0], [4.5, 2.3, 1.3, 0.3, 0],
            [4.4, 3.2, 1.3, 0.2, 0], [5.0, 3.5, 1.6, 0.6, 0], [5.1, 3.8, 1.9, 0.4, 0],
            [4.8, 3.0, 1.4, 0.3, 0], [5.1, 3.8, 1.6, 0.2, 0], [4.6, 3.2, 1.4, 0.2, 0],
            [5.3, 3.7, 1.5, 0.2, 0], [5.0, 3.3, 1.4, 0.2, 0],
            [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1],
            [5.5, 2.3, 4.0, 1.3, 1], [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1],
            [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1], [6.6, 2.9, 4.6, 1.3, 1],
            [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1],
            [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1],
            [6.7, 3.1, 4.4, 1.4, 1], [5.6, 3.0, 4.5, 1.5, 1], [5.8, 2.7, 4.1, 1.0, 1],
            [6.2, 2.2, 4.5, 1.5, 1], [5.6, 2.5, 3.9, 1.1, 1], [5.9, 3.2, 4.8, 1.8, 1],
            [6.1, 2.8, 4.0, 1.3, 1], [6.3, 2.5, 4.9, 1.5, 1], [6.1, 2.8, 4.7, 1.2, 1],
            [6.4, 2.9, 4.3, 1.3, 1], [6.6, 3.0, 4.4, 1.4, 1], [6.8, 2.8, 4.8, 1.4, 1],
            [6.7, 3.0, 5.0, 1.7, 1], [6.0, 2.9, 4.5, 1.5, 1], [5.7, 2.6, 3.5, 1.0, 1],
            [5.5, 2.4, 3.8, 1.1, 1], [5.5, 2.4, 3.7, 1.0, 1], [5.8, 2.7, 3.9, 1.2, 1],
            [6.0, 2.7, 5.1, 1.6, 1], [5.4, 3.0, 4.5, 1.5, 1], [6.0, 3.4, 4.5, 1.6, 1],
            [6.7, 3.1, 4.7, 1.5, 1], [6.3, 2.3, 4.4, 1.3, 1], [5.6, 3.0, 4.1, 1.3, 1],
            [5.5, 2.5, 4.0, 1.3, 1], [5.5, 2.6, 4.4, 1.2, 1], [6.1, 3.0, 4.6, 1.4, 1],
            [5.8, 2.6, 4.0, 1.2, 1], [5.0, 2.3, 3.3, 1.0, 1], [5.6, 2.7, 4.2, 1.3, 1],
            [5.7, 3.0, 4.2, 1.2, 1], [5.7, 2.9, 4.2, 1.3, 1], [6.2, 2.9, 4.3, 1.3, 1],
            [5.1, 2.5, 3.0, 1.1, 1], [5.7, 2.8, 4.1, 1.3, 1],
            [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2],
            [6.3, 2.9, 5.6, 1.8, 2], [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2],
            [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2], [6.7, 2.5, 5.8, 1.8, 2],
            [7.2, 3.6, 6.1, 2.5, 2], [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2],
            [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2],
            [6.4, 3.2, 5.3, 2.3, 2], [6.5, 3.0, 5.5, 1.8, 2], [7.7, 3.8, 6.7, 2.2, 2],
            [7.7, 2.6, 6.9, 2.3, 2], [6.0, 2.2, 5.0, 1.5, 2], [6.9, 3.2, 5.7, 2.3, 2],
            [5.6, 2.8, 4.9, 2.0, 2], [7.7, 2.8, 6.7, 2.0, 2], [6.3, 2.7, 4.9, 1.8, 2],
            [6.7, 3.3, 5.7, 2.1, 2], [7.2, 3.2, 6.0, 1.8, 2], [6.2, 2.8, 4.8, 1.8, 2],
            [6.1, 3.0, 4.9, 1.8, 2], [6.4, 2.8, 5.6, 2.1, 2], [7.2, 3.0, 5.8, 1.6, 2],
            [7.4, 2.8, 6.1, 1.9, 2], [7.9, 3.8, 6.4, 2.0, 2], [6.4, 2.8, 5.6, 2.2, 2],
            [6.3, 2.8, 5.1, 1.5, 2], [6.1, 2.6, 5.6, 1.4, 2], [7.7, 3.0, 6.1, 2.3, 2],
            [6.3, 3.4, 5.6, 2.4, 2], [6.4, 3.1, 5.5, 1.8, 2], [6.0, 3.0, 4.8, 1.8, 2],
            [6.9, 3.1, 5.4, 2.1, 2], [6.7, 3.1, 5.6, 2.4, 2], [6.9, 3.1, 5.1, 2.3, 2],
            [5.8, 2.7, 5.1, 1.9, 2], [6.8, 3.2, 5.9, 2.3, 2], [6.7, 3.3, 5.7, 2.5, 2],
            [6.7, 3.0, 5.2, 2.3, 2], [6.3, 2.5, 5.0, 1.9, 2], [6.5, 3.0, 5.2, 2.0, 2],
            [6.2, 3.4, 5.4, 2.3, 2], [5.9, 3.0, 5.1, 1.8, 2]
        ])
        
        self.X = iris_data[:, :4]
        self.y = iris_data[:, 4].astype(int)


class Blobs(Dataset):
    """
    Generate isotropic Gaussian blobs for clustering.
    
    Parameters
    ----------
    n_samples : int, default=100
        Total number of points to generate.
    n_features : int, default=2
        Number of features for each sample.
    centers : int or array-like, default=3
        Number of centers to generate, or fixed center locations.
    cluster_std : float, default=1.0
        Standard deviation of the clusters.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        Generated samples.
    y : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    """
    
    def __init__(self, n_samples=100, n_features=2, centers=3, 
                 cluster_std=1.0, random_state=None):
        """Initialize Blobs dataset."""
        super().__init__()
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std
        self.random_state = random_state
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate blob data."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Determine number of centers
        if isinstance(self.centers, int):
            n_centers = self.centers
            # Generate random centers
            centers = np.random.randn(n_centers, self.n_features) * 10
        else:
            centers = np.asarray(self.centers)
            n_centers = centers.shape[0]
        
        # Generate samples
        samples_per_center = self.n_samples // n_centers
        remainder = self.n_samples % n_centers
        
        X_list = []
        y_list = []
        
        for i in range(n_centers):
            # Number of samples for this center
            n = samples_per_center + (1 if i < remainder else 0)
            
            # Generate samples around this center
            samples = np.random.randn(n, self.n_features) * self.cluster_std + centers[i]
            labels = np.full(n, i)
            
            X_list.append(samples)
            y_list.append(labels)
        
        self.X = np.vstack(X_list)
        self.y = np.hstack(y_list).astype(int)
        
        # Shuffle
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
