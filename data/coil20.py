"""COIL-20 dataset."""

import numpy as np
from .base import register_dataset, normalize_features, create_dataloaders, compute_pca_embeddings


@register_dataset("coil20")
def load_coil20(config, with_embeddings=False, return_indices=False):
    """Load COIL-20 dataset (20 objects, 72 poses each, 128x128 grayscale)."""
    import os
    import zipfile
    import urllib.request
    from glob import glob
    from PIL import Image
    cache_dir = './data/raw/coil20'
    os.makedirs(cache_dir, exist_ok=True)
    
    zip_path = os.path.join(cache_dir, 'coil-20-proc.zip')
    extract_dir = os.path.join(cache_dir, 'coil-20-proc')
    
    # Download if needed
    if not os.path.exists(extract_dir):
        url = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
        print(f"Downloading COIL-20 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)
    
    # Load all PNG images
    image_files = sorted(glob(os.path.join(extract_dir, '*.png')))
    print(f"Found {len(image_files)} images")
    
    images, labels = [], []
    for fpath in image_files:
        fname = os.path.basename(fpath)
        # Parse obj{id}__*.png or obj{id}_*.png
        obj_id = int(fname.split('_')[0].replace('obj', '')) - 1
        img = np.array(Image.open(fpath).convert('L'), dtype=np.float32) / 255.0
        images.append(img)
        labels.append(obj_id)
    
    images = np.array(images)
    labels = np.array(labels, dtype=np.float32)
    print(f"Loaded {len(images)} images, {len(np.unique(labels))} classes")
    
    # Subsample if requested
    n_samples = config.get('n_samples', None)
    if n_samples is not None and n_samples < len(images):
        np.random.seed(config.get('seed', 42))
        idx = np.random.choice(len(images), n_samples, replace=False)
        images = images[idx]
        labels = labels[idx]
    
    seed = config.get('seed', 42)
    # 3-way split: ~70/15/15
    from .base import split_train_val_test
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
        images, labels, seed=seed
    )

    arch_type = config.get('arch_type', 'mlp')
    if arch_type == 'conv':
        train_data = train_data[:, np.newaxis, :, :]
        val_data = val_data[:, np.newaxis, :, :]
        test_data = test_data[:, np.newaxis, :, :]
    else:
        train_data = train_data.reshape(-1, 128 * 128)
        val_data = val_data.reshape(-1, 128 * 128)
        test_data = test_data.reshape(-1, 128 * 128)

    train_data, val_data, test_data = normalize_features(train_data, test_data, val_data=val_data)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_flat = train_data.reshape(len(train_data), -1)
        val_flat = val_data.reshape(len(val_data), -1)
        test_flat = test_data.reshape(len(test_data), -1)
        max_components = min(len(train_flat), train_flat.shape[1])
        if n_components > max_components:
            print(f"Warning: Reducing PCA components from {n_components} to {max_components}")
            n_components = max_components
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_flat, test_flat, n_components, seed=seed, val_data=val_flat
        )
        print(f"Computed PCA embeddings with {n_components} components")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )