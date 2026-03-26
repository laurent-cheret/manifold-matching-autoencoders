"""Evaluation metrics for autoencoder representations."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def distance_matrix(X):
    return squareform(pdist(X))


def neighbors_and_ranks(D, k):
    idx = np.argsort(D, axis=-1, kind='stable')
    return idx[:, 1:k+1], idx.argsort(axis=-1, kind='stable')


def trustworthiness(X, Z, k):
    n = X.shape[0]
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    Nx, Rx = neighbors_and_ranks(Dx, k)
    Nz, _ = neighbors_and_ranks(Dz, k)
    
    result = 0.0
    for i in range(n):
        for j in np.setdiff1d(Nz[i], Nx[i]):
            result += Rx[i, j] - k
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result


def continuity(X, Z, k):
    return trustworthiness(Z, X, k)


def rmse(X, Z):
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    n = X.shape[0]
    return np.sqrt(np.sum((Dx - Dz) ** 2) / n ** 2)


def distance_correlation(X, Z):
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    mask = np.triu(np.ones_like(Dx), k=1) > 0
    return np.corrcoef(Dx[mask], Dz[mask])[0, 1]


def knn_accuracy(latent, labels, k=10):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, latent, labels, cv=5)
    return scores.mean(), scores.std()


def triplet_accuracy(X, Z, n_triplets=10000, seed=42):
    """Compute triplet distance ranking accuracy."""
    np.random.seed(seed)
    n = X.shape[0]
    
    if n < 3:
        return 0.0
    
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    n_triplets = min(n_triplets, n * (n-1) * (n-2) // 6)
    
    correct = 0
    for _ in range(n_triplets):
        i, j, k = np.random.choice(n, 3, replace=False)
        x_order = Dx[i, j] < Dx[i, k]
        z_order = Dz[i, j] < Dz[i, k]
        if x_order == z_order:
            correct += 1
    
    return correct / n_triplets


def triplet_accuracy_batched(X, Z, batch_size=500, n_runs=10, seed=42):
    """Compute triplet accuracy on batched subsamples."""
    np.random.seed(seed)
    n = X.shape[0]
    
    if n <= batch_size:
        acc = triplet_accuracy(X, Z, n_triplets=10000, seed=seed)
        return acc, 0.0
    
    accs = []
    for run in range(n_runs):
        idx = np.random.choice(n, batch_size, replace=False)
        acc = triplet_accuracy(X[idx], Z[idx], n_triplets=10000, seed=seed+run)
        accs.append(acc)
    
    return np.mean(accs), np.std(accs)


def mrre(X, Z, k):
    """Mean Relative Rank Error."""
    n = X.shape[0]
    Dx, Dz = distance_matrix(X), distance_matrix(Z)
    
    Nx, Rx = neighbors_and_ranks(Dx, k)
    Nz, Rz = neighbors_and_ranks(Dz, k)
    
    mrre_zx = 0.0
    for i in range(n):
        for j in Nz[i]:
            mrre_zx += abs(Rx[i, j] - Rz[i, j]) / max(Rz[i, j], 1)
    
    mrre_xz = 0.0
    for i in range(n):
        for j in Nx[i]:
            mrre_xz += abs(Rx[i, j] - Rz[i, j]) / max(Rx[i, j], 1)
    
    C = n * sum(abs(2*j - n - 1) / j for j in range(1, k+1))
    
    return mrre_zx / C, mrre_xz / C


# ============== Density KL Divergence ==============

def density_kl_divergence(X, Z, sigma=0.1):
    """
    Compute KL divergence between density estimates.
    From TopoAE paper - measures local density preservation.
    """
    Dx = distance_matrix(X)
    Dz = distance_matrix(Z)
    
    # Normalize distances
    Dx = Dx / (Dx.max() + 1e-10)
    Dz = Dz / (Dz.max() + 1e-10)
    
    # Compute density estimates using Gaussian kernel
    density_x = np.sum(np.exp(-(Dx ** 2) / sigma), axis=-1)
    density_x = density_x / (density_x.sum() + 1e-10)
    
    density_z = np.sum(np.exp(-(Dz ** 2) / sigma), axis=-1)
    density_z = density_z / (density_z.sum() + 1e-10)
    
    # KL divergence
    eps = 1e-10
    kl = np.sum(density_x * (np.log(density_x + eps) - np.log(density_z + eps)))
    
    return kl

def density_kl_divergence_subsampled(X, Z, sigma=0.1, batch_size=500, n_runs=3, seed=42):
    """
    KL divergence between density estimates, with subsampling like persistence_wasserstein.
    This stabilizes the metric for large datasets.
    """
    np.random.seed(seed)
    n = X.shape[0]

    actual_batch = min(batch_size, n)
    actual_runs = 1 if n <= batch_size else n_runs

    kl_values = []

    for run in range(actual_runs):
        # Subsampling
        if n > batch_size:
            ids = np.random.choice(n, batch_size, replace=False)
            Xb, Zb = X[ids], Z[ids]
        else:
            Xb, Zb = X, Z

        # Distances
        Dx = distance_matrix(Xb)
        Dz = distance_matrix(Zb)

        # Normalize distances
        Dx = Dx / (Dx.max() + 1e-10)
        Dz = Dz / (Dz.max() + 1e-10)

        # Density estimates (Gaussian kernel)
        density_x = np.sum(np.exp(-(Dx ** 2) / sigma), axis=-1)
        density_x = density_x / (density_x.sum() + 1e-10)

        density_z = np.sum(np.exp(-(Dz ** 2) / sigma), axis=-1)
        density_z = density_z / (density_z.sum() + 1e-10)

        # KL divergence
        eps = 1e-10
        kl = np.sum(density_x * (np.log(density_x + eps) - np.log(density_z + eps)))
        kl_values.append(kl)

    return np.mean(kl_values)#, np.std(kl_values)


# ============== Wasserstein Distance on Persistence Diagrams ==============

def compute_persistence_diagram(X, max_dim=1, max_edge=np.inf):
    """Compute persistence diagram using gudhi."""
    try:
        import gudhi as gd
        D = distance_matrix(X)
        rips = gd.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()
        
        diagrams = {}
        for dim in range(max_dim + 1):
            intervals = st.persistence_intervals_in_dimension(dim)
            finite = intervals[np.isfinite(intervals[:, 1])] if len(intervals) > 0 else np.array([]).reshape(0, 2)
            diagrams[dim] = finite
        return diagrams
    except ImportError:
        return None


def wasserstein_distance_diagrams(dgm1, dgm2, p=1):
    """Compute Wasserstein distance between persistence diagrams."""
    try:
        import gudhi.wasserstein as gw
        return gw.wasserstein_distance(dgm1, dgm2, order=p)
    except ImportError:
        try:
            import gudhi.hera as hera
            return hera.wasserstein_distance(dgm1, dgm2, internal_p=p)
        except ImportError:
            return None


def persistence_wasserstein(X, Z, max_dim=1, batch_size=500, n_runs=3, seed=42):
    """Compute Wasserstein distance between persistence diagrams of X and Z."""
    np.random.seed(seed)
    n = X.shape[0]
    
    results = {dim: [] for dim in range(max_dim + 1)}
    
    actual_batch = min(batch_size, n)
    actual_runs = 1 if n <= batch_size else n_runs
    
    for run in range(actual_runs):
        if n > batch_size:
            ids = np.random.choice(n, batch_size, replace=False)
            X_batch, Z_batch = X[ids], Z[ids]
        else:
            X_batch, Z_batch = X, Z
        
        # Normalize for comparable scales
        X_dists = distance_matrix(X_batch)
        Z_dists = distance_matrix(Z_batch)
        X_norm = X_batch / (np.percentile(X_dists, 90) + 1e-10)
        Z_norm = Z_batch / (np.percentile(Z_dists, 90) + 1e-10)
        
        dgm_x = compute_persistence_diagram(X_norm, max_dim=max_dim)
        dgm_z = compute_persistence_diagram(Z_norm, max_dim=max_dim)
        
        if dgm_x is None or dgm_z is None:
            return None
        
        for dim in range(max_dim + 1):
            w = wasserstein_distance_diagrams(dgm_x[dim], dgm_z[dim])
            if w is not None:
                results[dim].append(w)
    
    return {dim: (np.mean(vals), np.std(vals)) if vals else (None, None) 
            for dim, vals in results.items()}

def clustering_ari_nmi(latent, labels):
    """
    Evaluate clustering quality using KMeans.
    
    Returns ARI and NMI which measure how well latent space clusters
    align with true cell types/classes.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latent)
    
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels, average_method='arithmetic')
    
    return float(ari), float(nmi)


def cluster_purity(latent, labels):
    """Compute cluster purity score."""
    from sklearn.cluster import KMeans
    
    n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latent)
    
    total_correct = 0
    for cluster_id in range(n_clusters):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        cluster_labels = labels[mask].astype(int)
        majority_count = np.bincount(cluster_labels).max()
        total_correct += majority_count
    
    return float(total_correct / len(labels))


def silhouette(latent, labels):
    """Compute silhouette score using ground truth labels."""
    from sklearn.metrics import silhouette_score
    
    n_unique = len(np.unique(labels))
    if n_unique < 2 or len(latent) <= n_unique:
        return None
    
    return float(silhouette_score(latent, labels, metric='euclidean'))


# Metric families for dynamic routing during HPO
_WASSERSTEIN_METRICS = {'wasserstein_H0', 'wasserstein_H1'}
_DENSITY_KL_METRICS = {'density_kl_0_01', 'density_kl_0_1', 'density_kl_1_0'}
_NEIGHBORHOOD_METRICS = {
    'trustworthiness', 'continuity', 'mrre_zx', 'mrre_xz',
    'distance_correlation', 'rmse', 'triplet_accuracy',
}
_CLUSTERING_METRICS = {
    'clustering_ari', 'clustering_nmi', 'cluster_purity', 'silhouette_score',
    'knn_accuracy_5', 'knn_accuracy_10',
}


def evaluate(X, Z, labels, ks=[10, 50, 100], compute_wasserstein=True, opt_metric=None):
    """Comprehensive evaluation of embedding quality.

    Args:
        X: original high-dimensional data (flattened)
        Z: latent/embedded representation
        labels: class labels
        ks: k values for neighborhood metrics
        compute_wasserstein: whether to compute expensive Wasserstein topology distance
        opt_metric: if set, only compute this metric family (fast HPO mode).
            Skips all unrelated expensive computations.

    Returns:
        dict of metrics (all values are Python floats)
    """
    results = {}

    # Determine which metric families to compute
    need_wasserstein = compute_wasserstein
    need_density_kl = True
    need_neighborhood = True
    need_clustering = True

    if opt_metric is not None:
        # Fast HPO mode: only compute the requested family + lightweight basics
        need_wasserstein = opt_metric in _WASSERSTEIN_METRICS
        need_density_kl = opt_metric in _DENSITY_KL_METRICS
        need_neighborhood = opt_metric in _NEIGHBORHOOD_METRICS or opt_metric.startswith('trustworthiness')
        need_clustering = opt_metric in _CLUSTERING_METRICS
        # Always compute distance_correlation as a cheap sanity check
        results['distance_correlation'] = float(distance_correlation(X, Z))
    else:
        results['distance_correlation'] = float(distance_correlation(X, Z))
        results['rmse'] = float(rmse(X, Z))

    # Neighborhood metrics
    if need_neighborhood:
        if opt_metric is None:
            pass  # rmse already computed above
        else:
            results['rmse'] = float(rmse(X, Z))

        for k in ks:
            if k < X.shape[0]:
                results[f'trustworthiness_{k}'] = float(trustworthiness(X, Z, k))
                results[f'continuity_{k}'] = float(continuity(X, Z, k))
                mrre_zx, mrre_xz = mrre(X, Z, k)
                results[f'mrre_zx_{k}'] = float(mrre_zx)
                results[f'mrre_xz_{k}'] = float(mrre_xz)

        trip_mean, trip_std = triplet_accuracy_batched(X, Z)
        results['triplet_accuracy'] = float(trip_mean)
        results['triplet_accuracy_std'] = float(trip_std)

    # kNN + clustering metrics
    if need_clustering:
        for k in [5, 10]:
            if k < X.shape[0]:
                m, s = knn_accuracy(Z, labels, k)
                results[f'knn_accuracy_{k}'] = float(m)
                results[f'knn_accuracy_{k}_std'] = float(s)
        ari, nmi = clustering_ari_nmi(Z, labels)
        results['clustering_ari'] = ari
        results['clustering_nmi'] = nmi
        results['cluster_purity'] = cluster_purity(Z, labels)
        sil = silhouette(Z, labels)
        if sil is not None:
            results['silhouette_score'] = sil

    # Density KL divergence at multiple scales
    if need_density_kl:
        for sigma in [0.01, 0.1, 1.0]:
            sigma_str = str(sigma).replace('.', '_')
            results[f'density_kl_{sigma_str}'] = float(
                density_kl_divergence_subsampled(X, Z, sigma=sigma)
            )

    # Wasserstein distance on persistence diagrams (most expensive)
    if need_wasserstein:
        wass = persistence_wasserstein(X, Z, max_dim=1, batch_size=500, n_runs=3)
        if wass is not None:
            for dim, (mean_val, std_val) in wass.items():
                if mean_val is not None:
                    results[f'wasserstein_H{dim}'] = float(mean_val)
                    results[f'wasserstein_H{dim}_std'] = float(std_val)

    return results