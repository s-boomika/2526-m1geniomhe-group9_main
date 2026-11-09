"""
Example: Decision Tree Classification on Iris Dataset

This example demonstrates how to use the DecisionTreeClassifier
from the upsaclay_ml library on the famous Iris dataset.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from upsaclay_ml.models.tree import DecisionTreeClassifier
from upsaclay_ml.datasets import Iris
from upsaclay_ml.model_selection import train_test_split, cross_val_score
from upsaclay_ml.metrics import accuracy_score


def main():
    """Run Decision Tree classification example."""
    print("=" * 60)
    print("Decision Tree Classification on Iris Dataset")
    print("=" * 60)
    print()
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = Iris()
    X, y = iris.X, iris.y
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target names: {iris.target_names}")
    print(f"Feature names: {iris.feature_names}")
    print()
    print(iris)
    print()
    
    # Split data
    print("Splitting data into train (75%) and test (25%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print()
    
    # Train Decision Tree
    print("Training Decision Tree (max_depth=3, criterion='gini')...")
    clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
    clf.fit(X_train, y_train)
    print("Training completed!")
    print()
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print()
    
    # Show feature importances
    print("Feature Importances:")
    print("-" * 50)
    for i, (name, importance) in enumerate(zip(iris.feature_names, clf.feature_importances_)):
        bar = '█' * int(importance * 50)
        print(f"{name:25s} | {bar} {importance:.3f}")
    print("-" * 50)
    print()
    
    # Cross-validation
    print("Performing 5-fold cross-validation...")
    clf_cv = DecisionTreeClassifier(max_depth=3, criterion='gini')
    cv_scores = cross_val_score(clf_cv, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    print()
    
    # Compare different max_depth values
    print("Comparing different max_depth values...")
    print("-" * 50)
    print(f"{'max_depth':>10s} | {'Train Acc':>10s} | {'Test Acc':>9s}")
    print("-" * 50)
    
    for depth in [1, 2, 3, 5, 10, None]:
        depth_str = str(depth) if depth is not None else "None"
        clf_d = DecisionTreeClassifier(max_depth=depth)
        clf_d.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf_d.predict(X_train))
        test_acc = accuracy_score(y_test, clf_d.predict(X_test))
        
        print(f"{depth_str:>10s} | {train_acc:10.3f} | {test_acc:9.3f}")
    
    print("-" * 50)
    print()
    
    # Confusion matrix
    print("Confusion Matrix:")
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(y_test, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    
    print("              Predicted")
    print("           ", end="")
    for i in range(3):
        print(f"{iris.target_names[i]:>12s}", end="")
    print()
    print("Actual")
    for i in range(3):
        print(f"{iris.target_names[i]:>10s}", end="")
        for j in range(3):
            print(f"{confusion_matrix[i, j]:12d}", end="")
        print()
    print()
    
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
