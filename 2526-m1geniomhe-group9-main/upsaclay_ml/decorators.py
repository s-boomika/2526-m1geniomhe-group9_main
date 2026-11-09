"""
Decorators for enhancing estimators and transformers.

This module provides various decorators to add functionality to models,
such as timing, validation, caching, and history tracking.
"""

import time
import functools
from collections import defaultdict
import pickle


def timed_fit(cls):
    """
    Decorator to measure and store the time taken to fit a model.
    
    Adds a `fit_time_` attribute to the class that stores the duration
    of the last fit operation in seconds.
    
    Example
    -------
    @timed_fit
    class MyModel(BaseEstimator):
        ...
    
    model = MyModel()
    model.fit(X, y)
    print(f"Fitting took {model.fit_time_:.2f} seconds")
    """
    original_fit = cls.fit
    
    @functools.wraps(original_fit)
    def timed_fit_wrapper(self, X, y=None):
        start_time = time.time()
        result = original_fit(self, X, y)
        end_time = time.time()
        self.fit_time_ = end_time - start_time
        return result
    
    cls.fit = timed_fit_wrapper
    return cls


def validate_params(**param_constraints):
    """
    Decorator to validate hyperparameters at initialization.
    
    Parameters
    ----------
    **param_constraints : dict
        Dictionary mapping parameter names to validation functions.
        Each validation function should take the parameter value and
        raise ValueError if invalid.
    
    Example
    -------
    @validate_params(
        n_clusters=lambda x: x > 0,
        max_iter=lambda x: x > 0
    )
    class KMeans(BaseEstimator):
        def __init__(self, n_clusters=3, max_iter=300):
            ...
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def validated_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)
            
            # Validate parameters
            for param_name, validator in param_constraints.items():
                if hasattr(self, param_name):
                    param_value = getattr(self, param_name)
                    try:
                        if not validator(param_value):
                            raise ValueError(
                                f"Invalid value for parameter '{param_name}': {param_value}"
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}': {e}"
                        )
        
        cls.__init__ = validated_init
        return cls
    
    return decorator


def cache_predictions(cls):
    """
    Decorator to cache predictions to avoid recomputing on same input.
    
    Adds a cache that stores predictions based on a hash of the input data.
    This can significantly speed up repeated predictions on the same data.
    
    Warning: Uses a lot of memory for large datasets.
    
    Example
    -------
    @cache_predictions
    class MyModel(BaseEstimator):
        ...
    """
    original_predict = cls.predict
    
    @functools.wraps(original_predict)
    def cached_predict(self, X):
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
        
        # Create a hash of the input
        import hashlib
        import numpy as np
        X_array = np.asarray(X)
        X_hash = hashlib.md5(X_array.tobytes()).hexdigest()
        
        # Check cache
        if X_hash in self._prediction_cache:
            return self._prediction_cache[X_hash]
        
        # Compute prediction
        result = original_predict(self, X)
        
        # Store in cache
        self._prediction_cache[X_hash] = result
        
        return result
    
    cls.predict = cached_predict
    return cls


def track_history(cls):
    """
    Decorator to track the history of fit operations.
    
    Adds a `history_` attribute that stores information about each
    fit operation, including timestamp and data shape.
    
    Example
    -------
    @track_history
    class MyModel(BaseEstimator):
        ...
    
    model = MyModel()
    model.fit(X1, y1)
    model.fit(X2, y2)
    print(model.history_)
    """
    original_fit = cls.fit
    
    @functools.wraps(original_fit)
    def history_tracking_fit(self, X, y=None):
        # Initialize history if it doesn't exist
        if not hasattr(self, 'history_'):
            self.history_ = []
        
        # Record fit event
        import numpy as np
        X_array = np.asarray(X)
        
        fit_info = {
            'timestamp': time.time(),
            'n_samples': X_array.shape[0],
            'n_features': X_array.shape[1] if X_array.ndim > 1 else 1,
            'has_target': y is not None
        }
        
        self.history_.append(fit_info)
        
        # Call original fit
        return original_fit(self, X, y)
    
    cls.fit = history_tracking_fit
    return cls


def frozen_model(cls):
    """
    Decorator to prevent modifications after the model is fitted.
    
    Once the model is fitted, attempting to fit it again will raise an error.
    This ensures the model's state is preserved.
    
    Example
    -------
    @frozen_model
    class MyModel(BaseEstimator):
        ...
    
    model = MyModel()
    model.fit(X, y)  # OK
    model.fit(X, y)  # Raises error
    """
    original_fit = cls.fit
    
    @functools.wraps(original_fit)
    def frozen_fit(self, X, y=None):
        if hasattr(self, '_is_frozen') and self._is_frozen:
            raise RuntimeError(
                f"{cls.__name__} is frozen and cannot be refitted. "
                "Create a new instance to train on different data."
            )
        
        result = original_fit(self, X, y)
        self._is_frozen = True
        return result
    
    cls.fit = frozen_fit
    return cls


def log_calls(cls):
    """
    Decorator to log all method calls with their arguments.
    
    Adds detailed logging of all method calls to help with debugging.
    
    Example
    -------
    @log_calls
    class MyModel(BaseEstimator):
        ...
    """
    import inspect
    
    # Get all methods
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue  # Skip private methods
        
        @functools.wraps(method)
        def logged_method(self, *args, _original_method=method, _method_name=name, **kwargs):
            print(f"[LOG] Calling {cls.__name__}.{_method_name}")
            print(f"      Args: {args}")
            print(f"      Kwargs: {kwargs}")
            
            result = _original_method(self, *args, **kwargs)
            
            print(f"[LOG] {cls.__name__}.{_method_name} completed")
            return result
        
        setattr(cls, name, logged_method)
    
    return cls


def serialize_model(filepath):
    """
    Decorator to add save/load functionality to a model.
    
    Adds `save()` and `load()` methods to serialize the model to disk.
    
    Parameters
    ----------
    filepath : str
        Default filepath for saving/loading the model.
    
    Example
    -------
    @serialize_model('model.pkl')
    class MyModel(BaseEstimator):
        ...
    
    model = MyModel()
    model.fit(X, y)
    model.save()  # Saves to 'model.pkl'
    
    loaded_model = MyModel.load()
    """
    def decorator(cls):
        def save(self, path=None):
            """Save the model to a file."""
            save_path = path or filepath
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved to {save_path}")
        
        @classmethod
        def load(cls, path=None):
            """Load a model from a file."""
            load_path = path or filepath
            with open(load_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {load_path}")
            return model
        
        cls.save = save
        cls.load = load
        return cls
    
    return decorator


def auto_numpy(cls):
    """
    Decorator to automatically convert inputs to numpy arrays.
    
    Wraps fit() and predict() methods to automatically convert
    list inputs to numpy arrays.
    
    Example
    -------
    @auto_numpy
    class MyModel(BaseEstimator):
        ...
    
    model = MyModel()
    model.fit([[1, 2], [3, 4]], [0, 1])  # Lists auto-converted to arrays
    """
    import numpy as np
    
    if hasattr(cls, 'fit'):
        original_fit = cls.fit
        
        @functools.wraps(original_fit)
        def numpy_fit(self, X, y=None):
            X = np.asarray(X)
            if y is not None:
                y = np.asarray(y)
            return original_fit(self, X, y)
        
        cls.fit = numpy_fit
    
    if hasattr(cls, 'predict'):
        original_predict = cls.predict
        
        @functools.wraps(original_predict)
        def numpy_predict(self, X):
            X = np.asarray(X)
            return original_predict(self, X)
        
        cls.predict = numpy_predict
    
    if hasattr(cls, 'transform'):
        original_transform = cls.transform
        
        @functools.wraps(original_transform)
        def numpy_transform(self, X):
            X = np.asarray(X)
            return original_transform(self, X)
        
        cls.transform = numpy_transform
    
    return cls
