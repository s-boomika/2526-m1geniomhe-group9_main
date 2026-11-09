"""
Clustering algorithms for unsupervised learning.

This module implements various clustering algorithms including K-Means,
Agglomerative Clustering, and Spectral Clustering.
"""

import numpy as np
from .unsupervised import UnsupervisedModel


class KMeans(UnsupervisedModel):
    """
    K-Means clustering algorithm.
    
    K-Means partitions n samples into k clusters by minimizing the within-cluster
    sum of squared distances to cluster centroids.
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int, optional
        Determines random number generation for centroid initialization.
        
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """Initialize K-Means with given parameters."""
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Attributes set during fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def fit(self, X, y=None):
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Initialize random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize centroids using k-means++ method
        self.cluster_centers_ = self._initialize_centroids(X)
        
        # Main k-means loop
        prev_inertia = np.inf
        
        for i in range(self.max_iter):
            # Assign samples to closest centroid
            self.labels_ = self._assign_labels(X)
            
            # Calculate new inertia
            self.inertia_ = self._calculate_inertia(X)
            
            # Check for convergence
            if prev_inertia - self.inertia_ < self.tol:
                self.n_iter_ = i + 1
                break
            
            prev_inertia = self.inertia_
            
            # Update centroids
            self._update_centroids(X)
        else:
            self.n_iter_ = self.max_iter
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        X = np.asarray(X)
        return self._assign_labels(X)
    
    def _initialize_centroids(self, X):
        """
        Initialize centroids using k-means++ algorithm.
        
        This method selects initial cluster centers that are generally far apart,
        which leads to better convergence.
        """
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]
        
        # Choose remaining centroids with probability proportional to distance squared
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) 
                                 for x in X])
            
            # Select next centroid with weighted probability
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(X[j])
                    break
        
        return np.array(centroids)
    
    def _assign_labels(self, X):
        """Assign each sample to the closest centroid."""
        distances = np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_] 
                             for x in X])
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X):
        """Update centroids as the mean of samples assigned to each cluster."""
        for k in range(self.n_clusters):
            cluster_samples = X[self.labels_ == k]
            if len(cluster_samples) > 0:
                self.cluster_centers_[k] = cluster_samples.mean(axis=0)
    
    def _calculate_inertia(self, X):
        """Calculate sum of squared distances to closest cluster center."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_samples = X[self.labels_ == k]
            if len(cluster_samples) > 0:
                inertia += np.sum((cluster_samples - self.cluster_centers_[k])**2)
        return inertia


class AgglomerativeClustering(UnsupervisedModel):
    """
    Agglomerative (hierarchical) clustering.
    
    Recursively merges the pair of clusters that minimally increases
    a given linkage criterion.
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    linkage : {'single', 'complete', 'average'}, default='average'
        Which linkage criterion to use.
        - 'single': minimum distance between all observations in pairs of clusters
        - 'complete': maximum distance between all observations in pairs of clusters
        - 'average': average distance between all observations in pairs of clusters
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
    n_clusters_ : int
        The number of clusters found by the algorithm.
    """
    
    def __init__(self, n_clusters=2, linkage='average'):
        """Initialize Agglomerative Clustering with given parameters."""
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        
        if linkage not in ['single', 'complete', 'average']:
            raise ValueError(f"Linkage must be 'single', 'complete', or 'average', got {linkage}")
        
        self.labels_ = None
        self.n_clusters_ = None
    
    def fit(self, X, y=None):
        """
        Fit the hierarchical clustering from features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Initialize each sample as its own cluster
        clusters = {i: [i] for i in range(n_samples)}
        
        # Build distance matrix
        dist_matrix = self._compute_distance_matrix(X)
        
        # Merge clusters until we have the desired number
        while len(clusters) > self.n_clusters:
            # Find the pair of clusters with minimum distance
            min_dist = np.inf
            merge_pair = None
            
            cluster_ids = list(clusters.keys())
            for i, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[i+1:]:
                    dist = self._cluster_distance(clusters[c1], clusters[c2], dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (c1, c2)
            
            # Merge the closest pair
            c1, c2 = merge_pair
            clusters[c1].extend(clusters[c2])
            del clusters[c2]
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, (cluster_id, sample_indices) in enumerate(clusters.items()):
            for idx in sample_indices:
                self.labels_[idx] = label
        
        self.n_clusters_ = len(clusters)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Note: For agglomerative clustering, we assign each new sample
        to the cluster with the nearest centroid from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        # This is a simplified prediction: assign to nearest cluster centroid
        # In practice, agglomerative clustering is typically only used in fit mode
        X = np.asarray(X)
        
        # Compute centroids from training data (stored implicitly via labels)
        # Note: We would need to store X_train for this, which is a design decision
        # For now, return the training labels (this is a limitation)
        raise NotImplementedError(
            "Predict mode for AgglomerativeClustering requires storing training data. "
            "Use fit_predict() instead, or use KMeans for prediction on new data."
        )
    
    def fit_predict(self, X, y=None):
        """
        Fit and return cluster labels for X.
        
        This is the preferred method for agglomerative clustering.
        """
        self.fit(X, y)
        return self.labels_
    
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _cluster_distance(self, cluster1, cluster2, dist_matrix):
        """
        Compute distance between two clusters based on linkage criterion.
        
        Parameters
        ----------
        cluster1, cluster2 : list of int
            Indices of samples in each cluster.
        dist_matrix : ndarray
            Pairwise distance matrix.
            
        Returns
        -------
        distance : float
            Distance between clusters according to linkage criterion.
        """
        distances = [dist_matrix[i, j] for i in cluster1 for j in cluster2]
        
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return np.mean(distances)


class SpectralClustering(UnsupervisedModel):
    """
    Spectral clustering using eigenvalue decomposition.
    
    Applies clustering to a projection of the normalized Laplacian.
    
    Parameters
    ----------
    n_clusters : int, default=2
        The dimension of the projection subspace.
    affinity : {'rbf', 'nearest_neighbors'}, default='rbf'
        How to construct the affinity matrix.
    gamma : float, default=1.0
        Kernel coefficient for rbf affinity.
    random_state : int, optional
        A pseudo random number generator seed.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix used for clustering.
    """
    
    def __init__(self, n_clusters=2, affinity='rbf', gamma=1.0, random_state=None):
        """Initialize Spectral Clustering with given parameters."""
        super().__init__()
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        
        self.labels_ = None
        self.affinity_matrix_ = None
    
    def fit(self, X, y=None):
        """
        Perform spectral clustering from features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        
        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity_matrix(X)
        
        # Compute normalized Laplacian
        laplacian = self._compute_normalized_laplacian(self.affinity_matrix_)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Use smallest k eigenvectors for clustering
        embedding = eigenvectors[:, :self.n_clusters]
        
        # Normalize rows to unit length
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        # Apply k-means to the embedding
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(embedding)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Note: Spectral clustering doesn't naturally support prediction on new data.
        Use fit_predict() instead.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        raise NotImplementedError(
            "Predict mode for SpectralClustering is not well-defined. "
            "Use fit_predict() instead."
        )
    
    def fit_predict(self, X, y=None):
        """
        Fit and return cluster labels for X.
        
        This is the preferred method for spectral clustering.
        """
        self.fit(X, y)
        return self.labels_
    
    def _compute_affinity_matrix(self, X):
        """
        Compute affinity (similarity) matrix.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        affinity : ndarray of shape (n_samples, n_samples)
            Affinity matrix.
        """
        n_samples = X.shape[0]
        
        if self.affinity == 'rbf':
            # RBF (Gaussian) kernel
            affinity = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i, n_samples):
                    dist_sq = np.sum((X[i] - X[j])**2)
                    similarity = np.exp(-self.gamma * dist_sq)
                    affinity[i, j] = similarity
                    affinity[j, i] = similarity
        else:
            raise ValueError(f"Affinity '{self.affinity}' not supported yet.")
        
        return affinity
    
    def _compute_normalized_laplacian(self, affinity):
        """
        Compute normalized graph Laplacian.
        
        Parameters
        ----------
        affinity : ndarray of shape (n_samples, n_samples)
            Affinity matrix.
            
        Returns
        -------
        laplacian : ndarray of shape (n_samples, n_samples)
            Normalized Laplacian matrix.
        """
        # Compute degree matrix
        degree = np.sum(affinity, axis=1)
        
        # Avoid division by zero
        degree_inv_sqrt = np.array([1.0 / np.sqrt(d) if d > 0 else 0 for d in degree])
        
        # Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        laplacian = np.eye(affinity.shape[0]) - D_inv_sqrt @ affinity @ D_inv_sqrt
        
        return laplacian
