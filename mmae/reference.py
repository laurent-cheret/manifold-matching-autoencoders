"""Compute the reference embedding used by the manifold-matching regularizer.

The reference is computed once on the training set and re-applied to val/test
data via `transform`. PCA supports out-of-sample transform natively. UMAP
exposes `.transform`. t-SNE does not, so we approximate val/test embeddings
with a fresh fit on the *concatenation* of train + val + test (the held-out
points are still untouched by the model — this only affects the reference).

Usage:
    ref_train, transformer = compute_reference(X_train, method='pca', dim=2)
    ref_val = transformer(X_val)   # may return None for t-SNE
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_reference(X_train, method="pca", dim=2, seed=42, X_eval=None):
    """Compute a reference embedding of `X_train`, optionally also for `X_eval`.

    Args:
        X_train: (N, D) array. Will be flattened to 2D if needed.
        method: 'pca' | 'umap' | 'tsne'.
        dim: target dimensionality of the reference embedding.
        seed: random seed.
        X_eval: optional list of additional arrays (e.g. [X_val, X_test])
            for which we want a reference embedding too.

    Returns:
        ref_train: (N, dim) np.float32 array.
        ref_eval:  list of (N_i, dim) np.float32 arrays, same length as X_eval
                   (None if X_eval is None).
    """
    X_train = np.asarray(X_train).reshape(len(X_train), -1).astype(np.float32)
    eval_arrays = None
    if X_eval is not None:
        eval_arrays = [np.asarray(a).reshape(len(a), -1).astype(np.float32) for a in X_eval]

    if method == "pca":
        n_comp = min(dim, X_train.shape[1], X_train.shape[0])
        model = PCA(n_components=n_comp, random_state=seed)
        ref_train = model.fit_transform(X_train).astype(np.float32)
        if dim > n_comp:
            # Pad with zeros if user asked for more components than feasible.
            ref_train = np.pad(ref_train, ((0, 0), (0, dim - n_comp)))
        ref_eval = None
        if eval_arrays is not None:
            ref_eval = []
            for a in eval_arrays:
                e = model.transform(a).astype(np.float32)
                if dim > n_comp:
                    e = np.pad(e, ((0, 0), (0, dim - n_comp)))
                ref_eval.append(e)
        return ref_train, ref_eval

    if method == "umap":
        try:
            import umap  # umap-learn
        except ImportError as e:
            raise ImportError(
                "UMAP reference requires `umap-learn`. Install with: pip install umap-learn"
            ) from e
        model = umap.UMAP(n_components=dim, random_state=seed)
        ref_train = model.fit_transform(X_train).astype(np.float32)
        ref_eval = None
        if eval_arrays is not None:
            ref_eval = [model.transform(a).astype(np.float32) for a in eval_arrays]
        return ref_train, ref_eval

    if method == "tsne":
        from sklearn.manifold import TSNE
        if eval_arrays is None:
            model = TSNE(n_components=dim, random_state=seed, init="pca")
            ref_train = model.fit_transform(X_train).astype(np.float32)
            return ref_train, None
        # t-SNE has no out-of-sample transform; embed train+eval jointly.
        sizes = [len(X_train)] + [len(a) for a in eval_arrays]
        joint = np.concatenate([X_train] + eval_arrays, axis=0)
        model = TSNE(n_components=dim, random_state=seed, init="pca")
        joint_emb = model.fit_transform(joint).astype(np.float32)
        offsets = np.cumsum([0] + sizes)
        ref_train = joint_emb[offsets[0]:offsets[1]]
        ref_eval = [joint_emb[offsets[i]:offsets[i + 1]] for i in range(1, len(sizes))]
        return ref_train, ref_eval

    raise ValueError(f"unknown reference method: {method!r}. Use 'pca', 'umap', or 'tsne'.")
