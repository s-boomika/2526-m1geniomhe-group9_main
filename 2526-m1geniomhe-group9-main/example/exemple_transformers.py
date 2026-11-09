"""
This script demonstrates the use of various data transformers from the
upsaclay_ml library, including StandardScaler, MinMaxScaler, Binarizer,
and OneHotEncoder.
"""

import numpy as np
from upsaclay_ml.datasets import Iris
from upsaclay_ml.transformers.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Binarizer,
    OneHotEncoder
)


def print_statistics(X, label="Data"):
    """
    Print statistical summary of data.
    
    Parameters
    ----------
    X : ndarray
        Data to summarize
    label : str
        Label for the data
    """
    print(f"\n    {label} Statistics:")
    print(f"    {'Metric':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print(f"    {'-'*72}")
    
    for i in range(min(X.shape[1], 4)):  # Show first 4 features
        feature_data = X[:, i]
        print(f"    {f'Feature {i}':<20} "
              f"{np.min(feature_data):>12.4f} "
              f"{np.max(feature_data):>12.4f} "
              f"{np.mean(feature_data):>12.4f} "
              f"{np.std(feature_data):>12.4f}")


def demo_standard_scaler():
    """
    Demonstrate StandardScaler on the Iris dataset.
    
    StandardScaler standardizes features by removing the mean and scaling
    to unit variance: z = (x - μ) / σ
    """
    print("=" * 70)
    print("STANDARDSCALER DEMONSTRATION")
    print("=" * 70)
    
    # Load Iris dataset
    print("\n[1] Loading Iris dataset...")
    iris = Iris()
    X = iris.X
    print(f"    Dataset shape: {X.shape}")
    print(f"    Feature names: {iris.feature_names}")
    
    # Display original statistics
    print("\n[2] Original data statistics:")
    print_statistics(X, "Original")
    
    # Apply StandardScaler
    print("\n[3] Applying StandardScaler...")
    scaler = StandardScaler()
    
    # Simulate train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # Fit on training data only
    print("\n[4] Fitting scaler on training data...")
    scaler.fit(X_train)
    print(f"    ✓ Learned means: {scaler.mean_}")
    print(f"    ✓ Learned stds (scale_): {scaler.scale_}")
    
    # Transform both train and test
    print("\n[5] Transforming training and test data...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print_statistics(X_train_scaled, "Scaled Training")
    print_statistics(X_test_scaled, "Scaled Test")
    
    # Verify properties
    print("\n[6] Verification:")
    print(f"    Training data mean ≈ 0: {np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)}")
    print(f"    Training data std ≈ 1: {np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)}")
    
    print("\n[7] Key Insights:")
    print("    • StandardScaler ensures zero mean and unit variance")
    print("    • Useful for algorithms sensitive to feature scale (SVM, KNN)")
    print("    • CRITICAL: Always fit on training data only to prevent data leakage")
    print("    • Test data uses training statistics for transformation")
    
    print("\n" + "=" * 70)
    print("✓ StandardScaler demonstration completed!")
    print("=" * 70 + "\n")


def demo_minmax_scaler():
    """
    Demonstrate MinMaxScaler on the Iris dataset.
    
    MinMaxScaler scales features to a given range (default [0, 1]):
    X_scaled = (X - X_min) / (X_max - X_min)
    """
    print("=" * 70)
    print("MINMAXSCALER DEMONSTRATION")
    print("=" * 70)
    
    # Load Iris dataset
    print("\n[1] Loading Iris dataset...")
    iris = Iris()
    X = iris.X
    
    # Display original statistics
    print("\n[2] Original data statistics:")
    print_statistics(X, "Original")
    
    # Apply MinMaxScaler with default range [0, 1]
    print("\n[3] Applying MinMaxScaler with range [0, 1]...")
    scaler_01 = MinMaxScaler(feature_range=(0, 1))
    scaler_01.fit(X)
    X_scaled_01 = scaler_01.transform(X)
    
    print_statistics(X_scaled_01, "Scaled [0, 1]")
    
    # Apply MinMaxScaler with custom range [-1, 1]
    print("\n[4] Applying MinMaxScaler with range [-1, 1]...")
    scaler_custom = MinMaxScaler(feature_range=(-1, 1))
    scaler_custom.fit(X)
    X_scaled_custom = scaler_custom.transform(X)
    
    print_statistics(X_scaled_custom, "Scaled [-1, 1]")
    
    # Verify properties
    print("\n[5] Verification:")
    print(f"    [0,1] range - Min ≈ 0: {np.allclose(np.min(X_scaled_01, axis=0), 0, atol=1e-10)}")
    print(f"    [0,1] range - Max ≈ 1: {np.allclose(np.max(X_scaled_01, axis=0), 1, atol=1e-10)}")
    print(f"    [-1,1] range - Min ≈ -1: {np.allclose(np.min(X_scaled_custom, axis=0), -1, atol=1e-10)}")
    print(f"    [-1,1] range - Max ≈ 1: {np.allclose(np.max(X_scaled_custom, axis=0), 1, atol=1e-10)}")
    
    # Compare sample transformation
    print("\n[6] Sample transformation (first sample, first feature):")
    original_val = X[0, 0]
    scaled_01_val = X_scaled_01[0, 0]
    scaled_custom_val = X_scaled_custom[0, 0]
    
    print(f"    Original value: {original_val:.4f}")
    print(f"    Scaled [0,1]: {scaled_01_val:.4f}")
    print(f"    Scaled [-1,1]: {scaled_custom_val:.4f}")
    
    print("\n[7] Key Insights:")
    print("    • MinMaxScaler bounds features to a specific range")
    print("    • Preserves the shape of the original distribution")
    print("    • Useful for algorithms needing bounded inputs (neural networks)")
    print("    • Sensitive to outliers (they can compress the majority of values)")
    
    print("\n" + "=" * 70)
    print("✓ MinMaxScaler demonstration completed!")
    print("=" * 70 + "\n")


def demo_binarizer():
    """
    Demonstrate Binarizer on synthetic data.
    
    Binarizer converts values to 0 or 1 based on a threshold.
    """
    print("=" * 70)
    print("BINARIZER DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic data with different value ranges
    print("\n[1] Creating synthetic data...")
    np.random.seed(42)
    X = np.random.randn(10, 3) * 10 + 5  # Mean=5, Std=10
    
    print(f"    Data shape: {X.shape}")
    print("\n    Original data (first 5 samples):")
    print("    " + "=" * 50)
    for i in range(5):
        print(f"    Sample {i}: [{X[i, 0]:>7.2f}, {X[i, 1]:>7.2f}, {X[i, 2]:>7.2f}]")
    
    # Apply Binarizer with default threshold (0.0)
    print("\n[2] Applying Binarizer with threshold=0.0...")
    binarizer_0 = Binarizer(threshold=0.0)
    binarizer_0.fit(X)
    X_binary_0 = binarizer_0.transform(X)
    
    print("\n    Binarized data (threshold=0.0, first 5 samples):")
    print("    " + "=" * 50)
    for i in range(5):
        print(f"    Sample {i}: [{X_binary_0[i, 0]:.0f}, {X_binary_0[i, 1]:.0f}, {X_binary_0[i, 2]:.0f}]")
    
    # Apply Binarizer with custom threshold (5.0)
    print("\n[3] Applying Binarizer with threshold=5.0...")
    binarizer_5 = Binarizer(threshold=5.0)
    binarizer_5.fit(X)
    X_binary_5 = binarizer_5.transform(X)
    
    print("\n    Binarized data (threshold=5.0, first 5 samples):")
    print("    " + "=" * 50)
    for i in range(5):
        print(f"    Sample {i}: [{X_binary_5[i, 0]:.0f}, {X_binary_5[i, 1]:.0f}, {X_binary_5[i, 2]:.0f}]")
    
    # Statistics
    print("\n[4] Binarization statistics:")
    print(f"    {'Threshold':<15} {'Ones (>threshold)':>20} {'Zeros (≤threshold)':>20}")
    print(f"    {'-'*58}")
    print(f"    {'0.0':<15} {np.sum(X_binary_0 == 1):>20} {np.sum(X_binary_0 == 0):>20}")
    print(f"    {'5.0':<15} {np.sum(X_binary_5 == 1):>20} {np.sum(X_binary_5 == 0):>20}")
    
    print("\n[5] Key Insights:")
    print("    • Binarizer creates binary features from continuous data")
    print("    • Values > threshold → 1, values ≤ threshold → 0")
    print("    • Useful for feature engineering (e.g., age > 18)")
    print("    • Stateless transformer (no parameters learned from data)")
    
    print("\n" + "=" * 70)
    print("✓ Binarizer demonstration completed!")
    print("=" * 70 + "\n")


def demo_onehot_encoder():
    """
    Demonstrate OneHotEncoder on categorical data.
    
    OneHotEncoder converts categorical variables into binary vectors.
    """
    print("=" * 70)
    print("ONEHOTENCODER DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic categorical data
    print("\n[1] Creating synthetic categorical data...")
    # Simulate: Color (0=Red, 1=Green, 2=Blue) and Size (0=Small, 1=Medium, 2=Large)
    X_categorical = np.array([
        [0, 0],  # Red, Small
        [1, 1],  # Green, Medium
        [2, 2],  # Blue, Large
        [0, 1],  # Red, Medium
        [1, 0],  # Green, Small
        [2, 1],  # Blue, Medium
        [0, 2],  # Red, Large
        [1, 2],  # Green, Large
    ])
    
    categories = {
        0: ['Red', 'Green', 'Blue'],
        1: ['Small', 'Medium', 'Large']
    }
    
    print(f"    Data shape: {X_categorical.shape}")
    print("\n    Original categorical data:")
    print(f"    {'Sample':<10} {'Color (0)':<15} {'Size (1)':<15}")
    print("    " + "=" * 42)
    for i, sample in enumerate(X_categorical):
        color = categories[0][sample[0]]
        size = categories[1][sample[1]]
        print(f"    {i:<10} {color:<15} {size:<15}")
    
    # Apply OneHotEncoder
    print("\n[2] Applying OneHotEncoder...")
    encoder = OneHotEncoder()
    encoder.fit(X_categorical)
    X_encoded = encoder.transform(X_categorical)
    
    print(f"    Original shape: {X_categorical.shape}")
    print(f"    Encoded shape: {X_encoded.shape}")
    print(f"    ✓ Created {X_encoded.shape[1]} binary features")
    
    # Show encoded data
    print("\n[3] Encoded data (one-hot representation):")
    print("    Original: [Color, Size]")
    print("    Encoded:  [R, G, B, S, M, L]")
    print("              (Red, Green, Blue, Small, Medium, Large)")
    print("    " + "=" * 60)
    
    for i in range(len(X_encoded)):
        original = X_categorical[i]
        encoded = X_encoded[i]
        color = categories[0][original[0]]
        size = categories[1][original[1]]
        encoded_str = ' '.join([f"{int(val)}" for val in encoded])
        print(f"    {i}: [{color:>5}, {size:>6}] → [{encoded_str}]")
    
    # Verify properties
    print("\n[4] Verification:")
    print(f"    Each row sums to 2: {np.all(np.sum(X_encoded, axis=1) == 2)}")
    print(f"    All values are 0 or 1: {np.all((X_encoded == 0) | (X_encoded == 1))}")
    print(f"    Number of categories per feature:")
    for feat_idx in range(X_categorical.shape[1]):
        n_categories = len(encoder.categories_[feat_idx])
        print(f"        Feature {feat_idx}: {n_categories} categories")
    
    print("\n[5] Key Insights:")
    print("    • OneHotEncoder converts categorical data to binary vectors")
    print("    • Each category becomes a separate binary feature")
    print("    • Prevents ordinal relationships in categorical data")
    print("    • Essential for algorithms that require numerical input")
    print("    • Warning: Can create high-dimensional data (curse of dimensionality)")
    
    print("\n" + "=" * 70)
    print("✓ OneHotEncoder demonstration completed!")
    print("=" * 70 + "\n")


def demo_transformer_pipeline():
    """
    Demonstrate combining multiple transformers in sequence.
    
    This shows how transformers can be chained together for
    comprehensive data preprocessing.
    """
    print("=" * 70)
    print("TRANSFORMER PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading Iris dataset...")
    iris = Iris()
    X = iris.X
    
    print(f"    Original data shape: {X.shape}")
    print_statistics(X[:10], "Original (first 10 samples)")
    
    # Step 1: MinMax scaling
    print("\n[2] Step 1: MinMax scaling to [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_step1 = scaler.fit_transform(X)
    print_statistics(X_step1[:10], "After MinMaxScaler")
    
    # Step 2: Binarization at 0.5 threshold
    print("\n[3] Step 2: Binarizing at threshold=0.5...")
    binarizer = Binarizer(threshold=0.5)
    X_step2 = binarizer.fit_transform(X_step1)
    
    print("\n    First 5 samples after binarization:")
    for i in range(5):
        print(f"    Sample {i}: {X_step2[i]}")
    
    # Summary
    print("\n[4] Transformation summary:")
    print(f"    Original range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"    After MinMax: [{X_step1.min():.2f}, {X_step1.max():.2f}]")
    print(f"    After Binarize: [{X_step2.min():.0f}, {X_step2.max():.0f}]")
    print(f"\n    Final data sparsity: {np.sum(X_step2 == 0) / X_step2.size * 100:.1f}% zeros")
    
    print("\n[5] Key Insights:")
    print("    • Transformers can be chained sequentially")
    print("    • Each transformer's output becomes the next one's input")
    print("    • Order matters: different sequences yield different results")
    print("    • Use Pipeline class for automatic chaining and state management")
    
    print("\n" + "=" * 70)
    print("✓ Transformer pipeline demonstration completed!")
    print("=" * 70 + "\n")


def main():
    """
    Main function to run all transformer demonstrations.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  UPSACLAY_ML TRANSFORMERS - COMPREHENSIVE DEMONSTRATION  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print("\nThis script demonstrates various data transformers available in")
    print("the upsaclay_ml library for feature engineering and preprocessing.\n")
    
    # Run all demonstrations
    demo_standard_scaler()
    input("Press Enter to continue to MinMaxScaler demonstration...")
    print("\n")
    
    demo_minmax_scaler()
    input("Press Enter to continue to Binarizer demonstration...")
    print("\n")
    
    demo_binarizer()
    input("Press Enter to continue to OneHotEncoder demonstration...")
    print("\n")
    
    demo_onehot_encoder()
    input("Press Enter to continue to Transformer Pipeline demonstration...")
    print("\n")
    
    demo_transformer_pipeline()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

