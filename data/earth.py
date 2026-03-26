"""Earth 3D dataset - continental land masses on a sphere."""

import numpy as np
import os

from .base import register_dataset, create_dataloaders, compute_pca_embeddings, split_train_val_test

NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"


def generate_earth_data(n_grid=200, seed=42):
    """Generate Earth point cloud using basemap and geopandas for accurate coastlines."""
    from mpl_toolkits.basemap import Basemap
    import geopandas
    from sklearn import preprocessing
    import pandas as pd
    
    # Download Natural Earth data
    cache_dir = "./data/natural_earth"
    os.makedirs(cache_dir, exist_ok=True)
    shp_path = os.path.join(cache_dir, "ne_110m_admin_0_countries.shp")
    
    if not os.path.exists(shp_path):
        import urllib.request
        import zipfile
        zip_path = os.path.join(cache_dir, "ne_110m_admin_0_countries.zip")
        print(f"Downloading Natural Earth data...")
        urllib.request.urlretrieve(NATURAL_EARTH_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)
    
    bm = Basemap(projection="cyl")
    
    xs, ys, zs = [], [], []
    phis, thetas = [], []
    
    for phi in np.linspace(-180, 180, num=n_grid):
        for theta in np.linspace(-90, 90, num=n_grid):
            if bm.is_land(phi, theta):
                phis.append(phi)
                thetas.append(theta)
                
                phi_rad = phi / 360 * 2 * np.pi
                theta_rad = theta / 360 * 2 * np.pi
                
                x = np.cos(phi_rad) * np.cos(theta_rad)
                y = np.cos(theta_rad) * np.sin(phi_rad)
                z = np.sin(theta_rad)
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
    
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    
    # Generate continent labels
    df = pd.DataFrame({"longitude": phis, "latitude": thetas})
    world = geopandas.read_file(shp_path)
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    results = geopandas.sjoin(gdf, world[['geometry', 'CONTINENT']], how="left")
    
    # Filter out Unknown (points where continent couldn't be determined)
    unknown_mask = results["CONTINENT"].isna()
    n_unknown = unknown_mask.sum()
    valid_mask = ~unknown_mask.values
    
    data = np.stack([xs[valid_mask], ys[valid_mask], zs[valid_mask]], axis=1).astype(np.float32)
    
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(results.loc[valid_mask, "CONTINENT"].values)
    labels = np.array(labels, dtype=np.float32)
    
    print(f"Continents: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"Filtered {n_unknown} unknown points")
    
    return data, labels, n_unknown


def load_earth_data(data_dir="./data", n_samples=None, seed=42):
    """Load or generate Earth 3D point cloud dataset."""
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "earth.npz")
    
    if os.path.exists(cache_path):
        loaded = np.load(cache_path)
        data = loaded['data']
        labels = loaded['labels']
        n_unknown = int(loaded.get('n_unknown', 0))
        print(f"Loaded cached Earth dataset: {len(data)} points (removed {n_unknown} unknown)")
    else:
        print("Generating Earth dataset (requires basemap and geopandas)...")
        data, labels, n_unknown = generate_earth_data(n_grid=300, seed=seed)
        np.savez(cache_path, data=data, labels=labels, n_unknown=n_unknown)
        print(f"Generated and cached Earth dataset: {len(data)} points (removed {n_unknown} unknown)")
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(data):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(data), n_samples, replace=False)
        data = data[idx]
        labels = labels[idx]
    
    return data, labels


@register_dataset("earth")
def load_earth(config, with_embeddings=False, return_indices=False):
    """Load Earth dataset."""
    data, labels = load_earth_data(
        n_samples=config.get("n_samples", 10000),
        seed=config.get("seed", 42)
    )
    
    seed = config.get("seed", 42)
    try:
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
            data, labels, seed=seed
        )
    except ValueError:
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_train_val_test(
            data, labels, seed=seed, stratify=False
        )

    # Center and scale uniformly (preserves spherical shape)
    mean = train_data.mean(axis=0)
    scale = np.abs(train_data - mean).max()
    train_data = ((train_data - mean) / scale).astype(np.float32)
    val_data = ((val_data - mean) / scale).astype(np.float32)
    test_data = ((test_data - mean) / scale).astype(np.float32)

    train_emb, val_emb, test_emb = None, None, None
    if with_embeddings:
        n_components = config.get("mmae_n_components", config.get("input_dim"))
        train_emb, val_emb, test_emb = compute_pca_embeddings(
            train_data, test_data, n_components, seed=seed, val_data=val_data
        )
        print(f"Computed PCA embeddings with {n_components} components")

    return create_dataloaders(
        train_data, val_data, test_data,
        train_labels, val_labels, test_labels,
        batch_size=config.get("batch_size", 64),
        train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
        return_indices=return_indices
    )