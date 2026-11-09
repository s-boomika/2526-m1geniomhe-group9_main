"""
This script demonstrates the use of various evaluation metrics from the
upsaclay_ml library, including classification metrics, regression metrics,
and clustering metrics.

"""

import numpy as np
from upsaclay_ml.datasets import Iris
from upsaclay_ml.models.tree import DecisionTreeClassifier
from upsaclay_ml.models.cluster import KMeans
from upsaclay_ml.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)


def demo_classification_metrics():
    """
    Demonstrate classification metrics on the Iris dataset.
    
    This function:
    1. Loads the Iris dataset
    2. Splits it into train/test sets
    3. Trains a Decision Tree classifier
    4. Evaluates using accuracy_score
    """
    print("=" * 70)
    print("CLASSIFICATION METRICS DEMONSTRATION")
    print("=" * 70)
    
    # Load Iris dataset
    print("\n[1] Loading Iris dataset...")
    iris = Iris()
    X, y = iris.X, iris.y
    print(f"    Dataset shape: {X.shape}")
    print(f"    Number of classes: {len(np.unique(y))}")
    
    # Split into train/test sets (70/30)
    print("\n[2] Splitting data into train/test sets (70/30)...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # Train Decision Tree classifier
    print("\n[3] Training Decision Tree classifier...")
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    clf.fit(X_train, y_train)
    print("    ✓ Model trained successfully")
    
    # Make predictions
    print("\n[4] Making predictions...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Evaluate using accuracy_score
    print("\n[5] Evaluating with accuracy_score...")
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n    {'Metric':<20} {'Train':>10} {'Test':>10}")
    print(f"    {'-'*42}")
    print(f"    {'Accuracy':<20} {train_accuracy:>10.4f} {test_accuracy:>10.4f}")
    
    # Per-class analysis
    print("\n[6] Per-class prediction distribution:")
    print(f"    {'Class':<15} {'True Count':>12} {'Predicted':>12}")
    print(f"    {'-'*42}")
    for cls in np.unique(y_test):
        true_count = np.sum(y_test == cls)
        pred_count = np.sum(y_pred_test == cls)
        print(f"    {f'Class {cls}':<15} {true_count:>12} {pred_count:>12}")
    
    print("\n" + "=" * 70)
    print("✓ Classification metrics demonstration completed!")
    print("=" * 70 + "\n")


def demo_regression_metrics():
    """
    Demonstrate regression metrics on synthetic data.
    
    This function:
    1. Generates synthetic regression data
    2. Creates predictions with varying levels of error
    3. Evaluates using MSE, MAE, and R² metrics
    """
    print("=" * 70)
    print("REGRESSION METRICS DEMONSTRATION")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic regression data...")
    np.random.seed(42)
    n_samples = 100
    
    # True underlying function: y = 3x + 2 + noise
    X = np.linspace(0, 10, n_samples)
    y_true = 3 * X + 2 + np.random.normal(0, 1, n_samples)
    
    print(f"    Samples generated: {n_samples}")
    print(f"    True function: y = 3x + 2 + ε, where ε ~ N(0, 1)")
    
    # Create predictions with different error levels
    print("\n[2] Creating predictions with varying error levels...")
    
    # Perfect model (with small noise)
    y_pred_good = 3 * X + 2 + np.random.normal(0, 0.5, n_samples)
    
    # Medium error model
    y_pred_medium = 2.5 * X + 3 + np.random.normal(0, 1.5, n_samples)
    
    # Poor model (wrong slope)
    y_pred_poor = 1.5 * X + 5 + np.random.normal(0, 2, n_samples)
    
    # Evaluate all models
    print("\n[3] Evaluating models with multiple regression metrics...")
    print(f"\n    {'Model':<15} {'MSE':>12} {'MAE':>12} {'R²':>12}")
    print(f"    {'-'*54}")
    
    models = [
        ("Good Model", y_pred_good),
        ("Medium Model", y_pred_medium),
        ("Poor Model", y_pred_poor)
    ]
    
    for model_name, y_pred in models:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"    {model_name:<15} {mse:>12.4f} {mae:>12.4f} {r2:>12.4f}")
    
    # Interpretation guide
    print("\n[4] Metric Interpretation Guide:")
    print("    • MSE (Mean Squared Error):")
    print("      - Lower is better (0 = perfect)")
    print("      - Penalizes large errors heavily (squared term)")
    print("\n    • MAE (Mean Absolute Error):")
    print("      - Lower is better (0 = perfect)")
    print("      - More robust to outliers than MSE")
    print("\n    • R² (Coefficient of Determination):")
    print("      - Range: (-∞, 1]")
    print("      - 1.0 = perfect fit")
    print("      - 0.0 = model same as predicting mean")
    print("      - < 0 = model worse than predicting mean")
    
    print("\n" + "=" * 70)
    print("✓ Regression metrics demonstration completed!")
    print("=" * 70 + "\n")


def demo_clustering_metrics():
    """
    Demonstrate clustering metrics (Silhouette Score) on Iris data.
    
    This function:
    1. Loads the Iris dataset
    2. Applies K-Means clustering with different k values
    3. Evaluates using Silhouette Score
    """
    print("=" * 70)
    print("CLUSTERING METRICS DEMONSTRATION")
    print("=" * 70)
    
    # Load Iris dataset (use only features, ignore labels)
    print("\n[1] Loading Iris dataset for unsupervised clustering...")
    iris = Iris()
    X = iris.X
    print(f"    Dataset shape: {X.shape}")
    print(f"    Using all 4 features for clustering")
    
    # Test different numbers of clusters
    print("\n[2] Testing K-Means with different cluster counts...")
    k_values = [2, 3, 4, 5]
    
    print(f"\n    {'K (clusters)':<15} {'Silhouette Score':>20} {'Interpretation':>20}")
    print(f"    {'-'*60}")
    
    best_k = None
    best_score = -1
    
    for k in k_values:
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100)
        labels = kmeans.fit_predict(X)
        
        # Calculate Silhouette Score
        score = silhouette_score(X, labels)
        
        # Interpretation
        if score > 0.7:
            interpretation = "Excellent"
        elif score > 0.5:
            interpretation = "Good"
        elif score > 0.25:
            interpretation = "Weak"
        else:
            interpretation = "Poor"
        
        print(f"    {k:<15} {score:>20.4f} {interpretation:>20}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\n[3] Best clustering configuration:")
    print(f"    Optimal K: {best_k} clusters")
    print(f"    Best Silhouette Score: {best_score:.4f}")
    
    # Detailed analysis for best k
    print(f"\n[4] Cluster distribution for K={best_k}:")
    kmeans = KMeans(n_clusters=best_k, random_state=42, max_iter=100)
    labels = kmeans.fit_predict(X)
    
    print(f"    {'Cluster':<15} {'Sample Count':>15}")
    print(f"    {'-'*32}")
    for cluster_id in range(best_k):
        count = np.sum(labels == cluster_id)
        print(f"    {f'Cluster {cluster_id}':<15} {count:>15}")
    
    # Silhouette Score interpretation
    print("\n[5] Silhouette Score Interpretation:")
    print("    • Range: [-1, 1]")
    print("    •  1.0: Clusters are perfectly separated")
    print("    •  0.0: Clusters are overlapping")
    print("    • -1.0: Samples assigned to wrong clusters")
    print("\n    Rule of thumb:")
    print("    • > 0.7: Strong structure")
    print("    • 0.5-0.7: Reasonable structure")
    print("    • 0.25-0.5: Weak structure")
    print("    • < 0.25: No substantial structure")
    
    print("\n" + "=" * 70)
    print("✓ Clustering metrics demonstration completed!")
    print("=" * 70 + "\n")


def main():
    """
    Main function to run all metric demonstrations.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  UPSACLAY_ML METRICS MODULE - COMPREHENSIVE DEMONSTRATION  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print("\nThis script demonstrates various evaluation metrics available in")
    print("the upsaclay_ml library across different machine learning tasks.\n")
    
    # Run all demonstrations
    demo_classification_metrics()
    input("Press Enter to continue to Regression Metrics demonstration...")
    print("\n")
    
    demo_regression_metrics()
    input("Press Enter to continue to Clustering Metrics demonstration...")
    print("\n")
    
    demo_clustering_metrics()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

