"""
Downstream task evaluation metrics for topology preservation.

These metrics demonstrate practical benefits of topology preservation:
- kNN classification accuracy: Do nearby points in latent space share labels?
- Clustering metrics (ARI, NMI): Does latent structure match true cell types?
- Silhouette score: Is the latent space well-clustered?
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    accuracy_score
)


# ============== kNN Classification ==============

def knn_classification_accuracy(latent, labels, k_values=[5, 10, 20], cv=5):
    """
    Evaluate kNN classification accuracy in latent space.
    
    Better topology preservation → neighbors in latent space share labels
    → higher kNN accuracy.
    
    Args:
        latent: (N, d) latent representations
        labels: (N,) ground truth labels
        k_values: list of k values to evaluate
        cv: number of cross-validation folds
    
    Returns:
        dict: {f'knn_k{k}': (mean_acc, std_acc) for each k}
    """
    results = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # Use stratified k-fold to handle class imbalance
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(knn, latent, labels, cv=skf, scoring='accuracy')
        
        results[f'knn_k{k}'] = scores.mean()
        results[f'knn_k{k}_std'] = scores.std()
    
    return results


# ============== Clustering Metrics ==============

def clustering_metrics(latent, labels, n_clusters=None, methods=['kmeans', 'hierarchical']):
    """
    Evaluate clustering quality in latent space.
    
    Better topology preservation → latent clusters align with true cell types
    → higher ARI and NMI.
    
    Args:
        latent: (N, d) latent representations
        labels: (N,) ground truth labels
        n_clusters: number of clusters (default: infer from labels)
        methods: clustering methods to evaluate
    
    Returns:
        dict: clustering metrics (ARI, NMI, silhouette)
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    results = {}
    
    for method in methods:
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            continue
        
        pred_labels = clusterer.fit_predict(latent)
        
        # Adjusted Rand Index: similarity of clusterings, adjusted for chance
        ari = adjusted_rand_score(labels, pred_labels)
        results[f'{method}_ari'] = ari
        
        # Normalized Mutual Information: information-theoretic measure
        nmi = normalized_mutual_info_score(labels, pred_labels, average_method='arithmetic')
        results[f'{method}_nmi'] = nmi
    
    # Silhouette score using ground truth labels (measures cluster separation)
    if len(np.unique(labels)) > 1 and len(latent) > len(np.unique(labels)):
        sil = silhouette_score(latent, labels, metric='euclidean')
        results['silhouette'] = sil
    
    return results


# ============== Cluster Purity ==============

def cluster_purity(latent, labels, n_clusters=None):
    """
    Compute cluster purity: fraction of points correctly assigned.
    
    For each predicted cluster, assign majority label, then compute accuracy.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latent)
    
    # For each cluster, find majority class
    total_correct = 0
    for cluster_id in range(n_clusters):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        cluster_labels = labels[mask]
        majority_count = np.bincount(cluster_labels.astype(int)).max()
        total_correct += majority_count
    
    purity = total_correct / len(labels)
    return purity


# ============== Label Propagation Accuracy ==============

def label_propagation_accuracy(latent, labels, train_ratio=0.1, k=15):
    """
    Semi-supervised evaluation: propagate labels using kNN.
    
    Simulates a realistic scenario where only a few cells are annotated.
    Better topology → better label propagation.
    """
    n = len(labels)
    n_train = int(n * train_ratio)
    
    # Stratified sampling of training indices
    np.random.seed(42)
    train_idx = []
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        n_label_train = max(1, int(len(label_idx) * train_ratio))
        selected = np.random.choice(label_idx, n_label_train, replace=False)
        train_idx.extend(selected)
    
    train_idx = np.array(train_idx)
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    
    # Train kNN on labeled subset
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(latent[train_idx], labels[train_idx])
    
    # Predict on unlabeled
    pred = knn.predict(latent[test_idx])
    acc = accuracy_score(labels[test_idx], pred)
    
    return acc


# ============== Combined Downstream Evaluation ==============

def evaluate_downstream_tasks(latent, labels, verbose=True):
    """
    Comprehensive downstream task evaluation.
    
    Returns all metrics that demonstrate practical benefits of 
    topology preservation for the ICML paper.
    """
    results = {}
    
    # kNN classification
    knn_results = knn_classification_accuracy(latent, labels, k_values=[5, 10, 20])
    results.update(knn_results)
    
    # Clustering metrics
    cluster_results = clustering_metrics(latent, labels)
    results.update(cluster_results)
    
    # Cluster purity
    results['cluster_purity'] = cluster_purity(latent, labels)
    
    # Label propagation (semi-supervised)
    results['label_prop_10pct'] = label_propagation_accuracy(latent, labels, train_ratio=0.1)
    results['label_prop_5pct'] = label_propagation_accuracy(latent, labels, train_ratio=0.05)
    
    if verbose:
        print("\n=== Downstream Task Evaluation ===")
        print(f"kNN Accuracy (k=10): {results.get('knn_k10', 0):.4f}")
        print(f"KMeans ARI: {results.get('kmeans_ari', 0):.4f}")
        print(f"KMeans NMI: {results.get('kmeans_nmi', 0):.4f}")
        print(f"Silhouette Score: {results.get('silhouette', 0):.4f}")
        print(f"Cluster Purity: {results.get('cluster_purity', 0):.4f}")
        print(f"Label Propagation (10%): {results.get('label_prop_10pct', 0):.4f}")
    
    return results


# ============== Topology-Downstream Correlation Analysis ==============

def analyze_topology_downstream_correlation(results_df):
    """
    Analyze correlation between topology metrics and downstream performance.
    
    This is key evidence for the paper: better topology → better downstream tasks.
    
    Args:
        results_df: DataFrame with columns for topology metrics 
                   (trustworthiness, continuity) and downstream metrics (knn, ari)
    
    Returns:
        dict: correlation coefficients and analysis
    """
    from scipy.stats import pearsonr, spearmanr
    
    topology_metrics = ['trustworthiness_avg', 'continuity_avg', 'distance_correlation']
    downstream_metrics = ['knn_k10', 'kmeans_ari', 'kmeans_nmi', 'cluster_purity']
    
    correlations = {}
    
    for topo in topology_metrics:
        if topo not in results_df.columns:
            continue
        for down in downstream_metrics:
            if down not in results_df.columns:
                continue
            
            # Remove NaN values
            mask = ~(results_df[topo].isna() | results_df[down].isna())
            if mask.sum() < 3:
                continue
            
            x = results_df[topo][mask]
            y = results_df[down][mask]
            
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            
            correlations[f'{topo}_vs_{down}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
    
    return correlations