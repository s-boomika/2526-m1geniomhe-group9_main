"""dataset loading utilities"""

from .base import Dataset, DataLoader
from .datasets import MNIST, Iris, Blobs

__all__ = [
    'Dataset',
    'DataLoader',
    'MNIST',
    'Iris',
    'Blobs',
]
