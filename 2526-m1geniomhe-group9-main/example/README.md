# Examples

This directory contains examples demonstrating the usage of the `upsaclay_ml` library.

## Available Examples

### 1. K-Means Clustering (`kmeans_example.py`)
Demonstrates K-Means clustering on a synthetic blobs dataset.

**Features:**
- Creating synthetic datasets
- Fitting K-Means models
- Evaluating with Silhouette score
- Visualization of results
- Comparing different numbers of clusters

**Run:**
```bash
cd examples
python kmeans_example.py
```

### 2. Decision Tree Classification (`decision_tree_example.py`)
Shows how to use the Decision Tree classifier on the Iris dataset.

**Features:**
- Loading built-in datasets
- Train-test splitting
- Model training and evaluation
- Feature importance analysis
- Cross-validation
- Confusion matrix

**Run:**
```bash
cd examples
python decision_tree_example.py
```

### 3. Pipeline (`pipeline_example.py`)
Demonstrates the Pipeline class for chaining transformers and models.

**Features:**
- Creating pipelines with multiple transformers
- Regression and classification examples
- Accessing and modifying pipeline steps
- Using dunder methods (__len__, __getitem__, __call__, etc.)

**Run:**
```bash
cd examples
python pipeline_example.py
```

## Requirements

To run the examples, you need:
- Python 3.7+
- numpy
- pandas
- matplotlib

Install dependencies:
```bash
pip install -r ../requirements.txt
```

## Expected Output

Each example will:
1. Print detailed progress and results to the console
2. Display evaluation metrics
3. Save visualizations (for applicable examples)

## Modifying Examples

Feel free to modify the examples to:
- Try different hyperparameters
- Use your own datasets
- Experiment with different transformers
- Compare multiple models
