"""
This script generates the two charts required for the report: this is a benchmark and evaluation Script for upsaclay_ml Library.
1. Computation Time Comparison (upsaclay_ml vs scikit-learn)
2. Confusion Matrix (Visual proof of Model Accuracy)
"""

import time
import numpy as np
import matplotlib.pyplot as plt

"""
Scikit-learn imports (Used ONLY for comparison/benchmarking) 
note again that it's not autorise for the project 
"""
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# importing our libraries 
from upsaclay_ml.models.cluster import KMeans as MyKMeans
from upsaclay_ml.models.tree import DecisionTreeClassifier as MyTree
# If you haven't implemented train_test_split, we can use numpy slicing
from upsaclay_ml.model_selection import train_test_split 

def run_time_benchmark():    
    """
    Generates Chart 1: Time Comparison (K-Means)
    """
    print("\n" + "="*50)
    print("Running Time Benchmark: K-Means (MyLib vs Sklearn)")
    print("="*50)
    
    # Define dataset sizes to test
    n_samples_list = [100, 500, 1000, 2000, 5000]
    my_times = []
    sklearn_times = []
    
    for n in n_samples_list:
        print(f"Processing dataset size N={n}...")
        
        # Generate synthetic data
        X, _ = make_blobs(n_samples=n, centers=3, n_features=10, random_state=42)
        
        # --- 1. Benchmark Scikit-learn (The Reference) ---
        start_sk = time.time()
        sk_model = SklearnKMeans(n_clusters=3, n_init=10, max_iter=300)
        sk_model.fit(X)
        end_sk = time.time()
        sklearn_times.append(end_sk - start_sk)
        
        # --- 2. Benchmark upsaclay_ml (Your Implementation) ---
        start_my = time.time()
        # Adjust arguments if your KMeans signature is different
        my_model = MyKMeans(n_clusters=3, max_iter=300) 
        my_model.fit(X)
        end_my = time.time()
        my_times.append(end_my - start_my)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_list, my_times, marker='o', linewidth=2, label='upsaclay_ml (Python/NumPy)')
    plt.plot(n_samples_list, sklearn_times, marker='x', linestyle='--', linewidth=2, label='scikit-learn (Optimized C)')
    
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Performance Benchmark: Training Time vs Dataset Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the chart
    filename = 'benchmark_time_comparison.png'
    plt.savefig(filename)
    print(f"\n[SUCCESS] Chart saved as: {filename}")
    plt.close()

def run_accuracy_visualization():
    """
    Generates Chart 2: Confusion Matrix (Decision Tree)
    """
    print("\n" + "="*50)
    print("Running Accuracy Evaluation: Decision Tree on Iris")
    print("="*50)
    
    # 1. Load Data
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names
    
    # 2. Split Data (Using your library's split if available)
    # If your train_test_split is not ready, you can use manual slicing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # 3. Train the model
    print("Fitting model...")
    clf = MyTree(max_depth=4, criterion='gini') # Adjust params as needed
    clf.fit(X_train, y_train)
    
    # 4. prediction
    y_pred = clf.predict(X_test)
    
    # 5. Calculate Confusion Matrix (Using sklearn metrics for pretty display)
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title('Confusion Matrix: Custom Decision Tree Evaluation', fontsize=14)
    
    # Save the chart
    filename = 'evaluation_confusion_matrix.png'
    plt.savefig(filename)
    print(f"\n[SUCCESS] Chart saved as: {filename}")
    plt.show() 

if __name__ == "__main__":
    # Run both benchmarks
    run_time_benchmark()
    run_accuracy_visualization()
    print("\nAll evaluations completed successfully.")
