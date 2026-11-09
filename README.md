# 2526-m1geniomhe-group*
# Machine Learning Library Built with OOP Principles


This is a machine learning library built from scratch using Object-Oriented Programming principles in Python. This library provides implementations of common ML algorithms, data transformers, and utilities following best practices and design patterns similar to scikit-learn.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Library Structure](#library-structure)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Class Diagrams](#class-diagrams)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/upsaclay_ml.git
cd upsaclay_ml

# Install the library (this installs dependencies automatically)
pip install .
```

### Development Mode

If you are developing the library and want changes to be reflected immediately:

```bash
pip install -e .
```

## Quick Start

### Example 1: K-Means Clustering

```python
from upsaclay_ml.models.cluster import KMeans
from upsaclay_ml.datasets import Blobs
from upsaclay_ml.metrics import silhouette_score

# Create synthetic dataset
blobs = Blobs(n_samples=200, centers=3)

# Fit K-Means
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(blobs.X)

# Evaluate
score = silhouette_score(blobs.X, labels)
print(f"Silhouette Score: {score:.3f}")
```

### Example 2: Pipeline with Transformers

```python
from upsaclay_ml.pipeline import Pipeline
from upsaclay_ml.models.linear_model import LinearRegression
from upsaclay_ml.transformers import StandardScaler

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Fit and predict
pipe.fit(X, y)
y_pred = pipe.predict(X_test)
```

## Library Structure

The library follows a modular design ensuring Clean Code and clear separation of concerns:

```text
upsaclay_ml/
├── models/              # Concrete implementations of algorithms
│   ├── cluster.py       # KMeans, Agglomerative, Spectral
│   ├── tree.py          # DecisionTreeClassifier
│   ├── linear_model.py  # LinearRegression
│   ├── supervised.py    # Base class for supervised learning
│   └── unsupervised.py  # Base class for unsupervised learning
├── transformers/        # Feature Engineering
│   └── preprocessing.py # StandardScaler, OneHotEncoder, etc.
├── datasets/            # Data Management
│   ├── base.py          # Dataset & DataLoader abstractions
│   └── datasets.py      # Built-in datasets (MNIST, Iris...)
├── pipeline/            # Workflow Orchestration
│   └── pipeline.py      # The Pipeline class (Composite Pattern)
├── utils/               # Internal Helper Functions
│   └── validation.py    # Input checks (check_array, etc.)
├── base.py              # Core Abstract Base Classes (BaseEstimator)
├── metrics.py           # Evaluation functions (accuracy, mse...)
├── model_selection.py   # Cross-validation utilities
└── decorators.py        # Python Decorators (@timed_fit, etc.)
```

## Core Components

### Models
*   **Supervised**: `DecisionTreeClassifier` (Gini/Entropy), `LinearRegression` (OLS).
*   **Unsupervised**: `KMeans` (Lloyd's algo + k-means++), `AgglomerativeClustering`, `SpectralClustering`.

### Transformers
*   `StandardScaler` (Z-score normalization)
*   `MinMaxScaler` (Range scaling)
*   `OneHotEncoder` (Categorical encoding)
*   `Binarizer` (Thresholding)

### Pipeline
*   Chains multiple `Transformers` and one `Estimator`.
*   Ensures consistent preprocessing between training and inference to prevent data leakage.

## Usage Examples

Check the `example/` directory for full scripts:

*   [`example/kmeans_example.py`](example/kmeans_example.py): Clustering demo with visualization.
*   [`example/decision_tree_example.py`](example/decision_tree_example.py): Classification on Iris dataset.
*   [`example/pipeline_example.py`](example/pipeline_example.py): Complex workflows showing polymorphism.

## Documentation

*   **[OOP report Group 9.pdf](docs/OOP%20report%20Group%209.pdf)**: detailed analysis of the architecture, mathematical foundations, and benchmarks.
*   **[UML Diagrams](docs/diagrams/)**: Visual blueprints of the system note that there are (V0, V1, Final) because of the continuous evolution of the project.

---

**Université d'Évry Paris-Saclay - M1 Geniomhe (Group 9)**:

**Mah Noor:**  
@mahnoor1022

**SAID CHANFI Sitty Nouzlat:**  
@moina-nouz

**SELVARADJOU Boomika:**  
@s-boomika
