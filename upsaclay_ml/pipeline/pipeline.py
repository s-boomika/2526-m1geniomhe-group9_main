"""
Pipeline for chaining transformers and estimators.
this module provides the Pipeline class which allows sequential application
of transformations followed by a final estimator.
"""

import numpy as np


class Pipeline:
    """
    Pipeline of transforms with a final estimator.
    
    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must implement fit and transform methods.
    The final estimator only needs to implement fit.
    
    The Pipeline automatically chains fit_transform → fit_transform → ... → fit
    on the training data, and transform → transform → ... → predict on test data.
    
    Parameters
    ----------
    steps : list of (name, transform) tuples
        List of (name, transform) tuples that are chained in sequential order.
        The last object is an estimator.
        
    Attributes
    ----------
    steps : list of tuples
        The list of (name, transform) tuples.
    named_steps : dict
        Dictionary allowing access to steps by name.
    """
    
    def __init__(self, steps):
        """
        Initialize Pipeline with a list of steps.
        
        Parameters
        ----------
        steps : list of tuples
            List of (name, transformer/estimator) tuples.
        """
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}
        
        # Validate steps
        self._validate_steps()
    
    def _validate_steps(self):
        """Validate that all steps (except last) have fit and transform methods."""
        if len(self.steps) == 0:
            raise ValueError("Pipeline must have at least one step.")
        
        # Check all intermediate steps have fit and transform
        for name, step in self.steps[:-1]:
            if not (hasattr(step, 'fit') and hasattr(step, 'transform')):
                raise TypeError(
                    f"All intermediate steps should implement fit and transform. "
                    f"'{name}' (type {type(step)}) doesn't."
                )
        
        # Check final step has fit
        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, 'fit'):
            raise TypeError(
                f"Last step should implement fit. "
                f"'{final_name}' (type {type(final_step)}) doesn't."
            )
    
    def fit(self, X, y=None):
        """
        Fit all transforms one after the other and fit the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        self : Pipeline
            This estimator.
        """
        X_transformed = X
        
        # Fit and transform through all intermediate steps
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)
        
        # Fit the final estimator
        final_name, final_step = self.steps[-1]
        final_step.fit(X_transformed, y)
        
        return self
    
    def predict(self, X):
        """
        Transform the data, and predict with the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
            
        Returns
        -------
        y_pred : array-like
            Predictions.
        """
        X_transformed = X
        
        # Transform through all intermediate steps
        for name, step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        
        # Predict with final estimator
        final_name, final_step = self.steps[-1]
        return final_step.predict(X_transformed)
    
    def fit_predict(self, X, y=None):
        """
        Fit the pipeline and predict on the same data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        y_pred : array-like
            Predictions on X.
        """
        self.fit(X, y)
        return self.predict(X)
    
    def __len__(self):
        """Return the number of steps in the pipeline."""
        return len(self.steps)
    
    def __getitem__(self, index):
        """
        Access a step by index or name.
        
        Parameters
        ----------
        index : int or str
            Index or name of the step.
            
        Returns
        -------
        step : object
            The transformer or estimator.
        """
        if isinstance(index, str):
            # Access by name
            if index not in self.named_steps:
                raise KeyError(f"Step '{index}' not found in pipeline.")
            return self.named_steps[index]
        elif isinstance(index, int):
            # Access by index
            if index < 0 or index >= len(self.steps):
                raise IndexError(f"Index {index} out of range for pipeline with {len(self.steps)} steps.")
            return self.steps[index][1]
        else:
            raise TypeError(f"Index must be int or str, got {type(index)}.")
    
    def __setitem__(self, index, value):
        """
        Set a step by index or name.
        
        Parameters
        ----------
        index : int or str
            Index or name of the step.
        value : tuple or object
            Either (name, step) tuple or just the step.
        """
        if isinstance(index, str):
            # Set by name
            if index not in self.named_steps:
                raise KeyError(f"Step '{index}' not found in pipeline.")
            
            # Find the index of this name
            for i, (name, _) in enumerate(self.steps):
                if name == index:
                    if isinstance(value, tuple):
                        self.steps[i] = value
                        self.named_steps = {name: step for name, step in self.steps}
                    else:
                        self.steps[i] = (name, value)
                        self.named_steps[name] = value
                    break
        elif isinstance(index, int):
            # Set by index
            if index < 0 or index >= len(self.steps):
                raise IndexError(f"Index {index} out of range for pipeline with {len(self.steps)} steps.")
            
            if isinstance(value, tuple):
                self.steps[index] = value
                self.named_steps = {name: step for name, step in self.steps}
            else:
                # Keep the same name, replace the step
                name = self.steps[index][0]
                self.steps[index] = (name, value)
                self.named_steps[name] = value
        else:
            raise TypeError(f"Index must be int or str, got {type(index)}.")
        
        # Re-validate steps
        self._validate_steps()
    
    def __iter__(self):
        """
        Iterate through pipeline steps.
        
        Yields
        ------
        (name, step) : tuple
            Name and object for each step.
        """
        for name, step in self.steps:
            yield (name, step)
    
    def __call__(self, X):
        """
        Shortcut for predict.
        
        Parameters
        ----------
        X : array-like
            Data to predict on.
            
        Returns
        -------
        y_pred : array-like
            Predictions.
        """
        return self.predict(X)
    
    def __repr__(self):
        """Return string representation of the pipeline."""
        step_strs = [f"('{name}', {step})" for name, step in self.steps]
        steps_repr = ",\n    ".join(step_strs)
        return f"Pipeline([\n    {steps_repr}\n])"
    
    def __str__(self):
        """Return human-readable string representation."""
        return self.__repr__()
    
    def __eq__(self, other):
        """
        Compare pipelines by steps.
        
        Parameters
        ----------
        other : Pipeline
            Another pipeline to compare with.
            
        Returns
        -------
        equal : bool
            True if pipelines have the same steps.
        """
        if not isinstance(other, Pipeline):
            return False
        
        if len(self.steps) != len(other.steps):
            return False
        
        for (name1, step1), (name2, step2) in zip(self.steps, other.steps):
            if name1 != name2 or step1 != step2:
                return False
        
        return True
