"""
Example: Complete Pipeline with Transformers and Model
this example demonstrates how to chain multiple transformers and a model
using the Pipeline class from the upsaclay_ml library.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from upsaclay_ml.pipeline import Pipeline
from upsaclay_ml.models.linear_model import LinearRegression
from upsaclay_ml.models.cluster import KMeans
from upsaclay_ml.transformers import StandardScaler, MinMaxScaler
from upsaclay_ml.datasets import Iris
from upsaclay_ml.model_selection import train_test_split
from upsaclay_ml.metrics import mean_squared_error, accuracy_score


def pipeline_regression_example():
    """Example of using Pipeline for regression."""
    print("=" * 60)
    print("Pipeline Example 1: Regression with Scaling")
    print("=" * 60)
    print()
    
    # Generate synthetic regression data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_coef = np.array([2.0, -1.5, 3.0])
    y = X @ true_coef + np.random.randn(100) * 0.1
    
    print(f"Dataset shape: {X.shape}")
    print(f"True coefficients: {true_coef}")
    print()
    
    # Create pipeline
    print("Creating pipeline: StandardScaler -> LinearRegression")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    print(f"Pipeline: {pipe}")
    print(f"Number of steps: {len(pipe)}")
    print()
    
    # Fit pipeline
    print("Fitting pipeline...")
    pipe.fit(X, y)
    print("Pipeline fitted!")
    print()
    
    # Make predictions
    y_pred = pipe(X)  # Using __call__ shortcut
    mse = mean_squared_error(y, y_pred)
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Learned coefficients: {pipe['model'].coef_}")
    print(f"Learned intercept: {pipe['model'].intercept_:.6f}")
    print()
    
    # Access individual steps
    print("Accessing pipeline steps:")
    print(f"  Step 0 (by index): {pipe[0]}")
    print(f"  Step 'scaler' (by name): {pipe['scaler']}")
    print()
    
    # Iterate through steps
    print("Iterating through pipeline steps:")
    for name, step in pipe:
        print(f"  {name}: {step.__class__.__name__}")
    print()


def pipeline_classification_example():
    """Example of using Pipeline for classification with Iris dataset."""
    print("=" * 60)
    print("Pipeline Example 2: Classification with Multiple Transformers")
    print("=" * 60)
    print()
    
    # Load Iris data (using first 2 features for simplicity)
    iris = Iris()
    X = iris.X[:, :2]  # Use only first 2 features
    y = iris.y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print()
    
    # Create a more complex pipeline
    print("Creating pipeline:")
    print("  1. MinMaxScaler (normalize features)")
    print("  2. StandardScaler (additional standardization)")
    print("  3. KMeans (clustering as surrogate for classification)")
    print()
    
    pipe = Pipeline([
        ('minmax', MinMaxScaler()),
        ('std_scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])
    
    # Fit and predict
    print("Fitting pipeline...")
    pipe.fit(X_train)
    
    print("Making predictions...")
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    # Note: KMeans cluster labels may not match class labels 
    # This is just for demonstration
    print()
    print(f"Training cluster distribution: {np.bincount(y_pred_train)}")
    print(f"Test cluster distribution: {np.bincount(y_pred_test)}")
    print()
    
    # Modify pipeline step
    print("Replacing KMeans with 4 clusters...")
    pipe['kmeans'] = KMeans(n_clusters=4, random_state=42)
    pipe.fit(X_train)
    y_pred_new = pipe.predict(X_test)
    print(f"New cluster distribution: {np.bincount(y_pred_new)}")
    print()


def pipeline_flexibility_example():
    """Demonstrate Pipeline flexibility and dunder methods."""
    print("=" * 60)
    print("Pipeline Example 3: Flexibility and Dunder Methods")
    print("=" * 60)
    print()
    
    # Create datasets
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create two pipelines
    pipe1 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KMeans(n_clusters=2, random_state=42))
    ])
    
    pipe2 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KMeans(n_clusters=2, random_state=42))
    ])
    
    pipe3 = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', KMeans(n_clusters=2, random_state=42))
    ])
    
    # Test equality
    print(f"pipe1 == pipe2: {pipe1 == pipe2}")
    print(f"pipe1 == pipe3: {pipe1 == pipe3}")
    print()
    
    # Test length
    print(f"len(pipe1): {len(pipe1)}")
    print()
    
    # Test repr
    print("Pipeline representation:")
    print(pipe1)
    print()
    
    # Test indexing and modification
    print("Accessing and modifying steps:")
    print(f"Original step 0: {pipe1[0]}")
    
    pipe1[0] = MinMaxScaler()
    print(f"Modified step 0: {pipe1[0]}")
    print()
    
    # Test iteration
    print("Iterating through pipeline:")
    for i, (name, step) in enumerate(pipe1):
        print(f"  Step {i}: {name} -> {step.__class__.__name__}")
    print()


def main():
    """Run all pipeline examples."""
    pipeline_regression_example()
    print("\n" * 2)
    
    pipeline_classification_example()
    print("\n" * 2)
    
    pipeline_flexibility_example()
    
    print("=" * 60)
    print("All pipeline examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
