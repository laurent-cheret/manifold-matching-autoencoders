"""MMAE Variants - Experimental approaches to manifold-aware distance matching."""

import torch
import torch.nn as nn
from .base import register_model, get_encoder, get_decoder


# =============================================================================
# Variant 1: Reconstruction-Based Distance (MMAE-Recon)
# =============================================================================
# Idea: Use decoder reconstruction error along interpolation paths as a proxy
# for manifold distance. High error = path crosses "off-manifold" regions.

def interpolation_distance(z, decoder, n_steps=5):
    """Estimate manifold distance via reconstruction difficulty along paths."""
    B = z.shape[0]
    device = z.device
    
    # Sample random pairs
    idx1 = torch.randperm(B, device=device)[:B//2]
    idx2 = torch.randperm(B, device=device)[:B//2]
    
    z1, z2 = z[idx1], z[idx2]
    
    # Interpolate and measure reconstruction consistency
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    max_errors = []
    for i in range(len(idx1)):
        # Interpolation path
        path = z1[i:i+1] * (1 - alphas.view(-1,1)) + z2[i:i+1] * alphas.view(-1,1)
        
        # Decode path
        with torch.no_grad():
            recons = decoder(path)
        
        # Re-encode and measure cycle consistency
        # (we'll need encoder passed in - simplified version below)
        # For now: use reconstruction variance as proxy
        recon_var = recons.var(dim=0).mean()
        max_errors.append(recon_var)
    
    return torch.stack(max_errors)


def recon_distance_loss(z, ref_emb, decoder, n_steps=5):
    """Match reconstruction-based distances to reference distances."""
    B = z.shape[0]
    device = z.device
    
    # Sample pairs
    n_pairs = min(B * 2, 128)
    idx1 = torch.randint(0, B, (n_pairs,), device=device)
    idx2 = torch.randint(0, B, (n_pairs,), device=device)
    
    z1, z2 = z[idx1], z[idx2]
    r1, r2 = ref_emb[idx1], ref_emb[idx2]
    
    # Reference distances
    ref_dist = (r1 - r2).norm(dim=1)
    
    # Latent Euclidean distances
    lat_dist = (z1 - z2).norm(dim=1)
    
    # Interpolation roughness as manifold distance proxy
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1, 1)
    paths = z1.unsqueeze(0) * (1 - alphas) + z2.unsqueeze(0) * alphas  # (n_steps, n_pairs, latent_dim)
    
    # Decode all points
    paths_flat = paths.view(-1, z.shape[-1])
    recons_flat = decoder(paths_flat)
    recons = recons_flat.view(n_steps, n_pairs, -1)
    
    # Path smoothness: consecutive reconstruction differences
    recon_diffs = (recons[1:] - recons[:-1]).norm(dim=-1)  # (n_steps-1, n_pairs)
    path_length = recon_diffs.sum(dim=0)  # (n_pairs,)
    
    # Normalize both
    lat_dist_norm = lat_dist / (lat_dist.max() + 1e-8)
    path_length_norm = path_length / (path_length.max() + 1e-8)
    ref_dist_norm = ref_dist / (ref_dist.max() + 1e-8)
    
    # Combined distance: Euclidean + path roughness
    combined_dist = 0.5 * lat_dist_norm + 0.5 * path_length_norm
    
    return ((combined_dist - ref_dist_norm) ** 2).mean()


@register_model('mmae_recon')
class MMAERecon(nn.Module):
    """MMAE with reconstruction-based distance estimation."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("mmae_lambda", 1.0)
        self.n_interp_steps = config.get("n_interp_steps", 5)
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, ref_emb=None):
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        if ref_emb is not None:
            dist_loss = recon_distance_loss(z, ref_emb, self.decoder, self.n_interp_steps)
            total_loss = rec_loss + self.lam * dist_loss
            return total_loss, {
                "recon_loss": rec_loss.item(),
                "dist_loss": dist_loss.item()
            }
        else:
            return rec_loss, {"recon_loss": rec_loss.item()}


# =============================================================================
# Variant 2: Rank Preservation (MMAE-Rank)
# =============================================================================
# Idea: Preserve distance ORDERING rather than distances themselves.
# "If A closer to B than to C in input, same should hold in latent"
# This allows stretching/unfolding while maintaining topology.

def rank_loss(z, ref_emb, n_triplets=256):
    """Triplet-based rank preservation loss."""
    B = z.shape[0]
    device = z.device
    
    if B < 3:
        return torch.tensor(0.0, device=device)
    
    # Sample triplets (anchor, positive, negative based on ref distances)
    n_triplets = min(n_triplets, B * 3)
    
    anchors = torch.randint(0, B, (n_triplets,), device=device)
    others1 = torch.randint(0, B, (n_triplets,), device=device)
    others2 = torch.randint(0, B, (n_triplets,), device=device)
    
    # Reference distances
    ref_d1 = (ref_emb[anchors] - ref_emb[others1]).norm(dim=1)
    ref_d2 = (ref_emb[anchors] - ref_emb[others2]).norm(dim=1)
    
    # Latent distances
    lat_d1 = (z[anchors] - z[others1]).norm(dim=1)
    lat_d2 = (z[anchors] - z[others2]).norm(dim=1)
    
    # Soft rank loss: if ref_d1 < ref_d2, then lat_d1 should be < lat_d2
    # Using soft margin: max(0, lat_d1 - lat_d2 + margin) when ref_d1 < ref_d2
    margin = 0.1
    
    # Which pairs have d1 < d2 in reference?
    ref_order = (ref_d1 < ref_d2).float()  # 1 if d1 < d2, else 0
    
    # Violation: d1 < d2 in ref but d1 > d2 in latent (and vice versa)
    lat_diff = lat_d1 - lat_d2  # positive if d1 > d2
    
    # Loss when ref says d1 < d2: penalize if lat_d1 > lat_d2
    loss_case1 = ref_order * torch.relu(lat_diff + margin)
    # Loss when ref says d1 > d2: penalize if lat_d1 < lat_d2
    loss_case2 = (1 - ref_order) * torch.relu(-lat_diff + margin)
    
    return (loss_case1 + loss_case2).mean()


def soft_rank_correlation_loss(z, ref_emb):
    """Spearman-like rank correlation on distances."""
    z_dist = torch.cdist(z, z, p=2)
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
    mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
    z_flat = z_dist[mask]
    ref_flat = ref_dist[mask]
    
    # Differentiable ranking via softmax
    temp = 0.1
    z_ranks = (z_flat.unsqueeze(1) > z_flat.unsqueeze(0)).float().sum(dim=1)
    ref_ranks = (ref_flat.unsqueeze(1) > ref_flat.unsqueeze(0)).float().sum(dim=1)
    
    # Normalize ranks
    n = z_ranks.shape[0]
    z_ranks = z_ranks / n
    ref_ranks = ref_ranks / n
    
    # Rank correlation
    z_c = z_ranks - z_ranks.mean()
    ref_c = ref_ranks - ref_ranks.mean()
    
    corr = (z_c * ref_c).sum() / (
        torch.sqrt((z_c**2).sum() * (ref_c**2).sum()) + 1e-8
    )
    
    return 1 - corr


@register_model('mmae_rank')
class MMAERank(nn.Module):
    """MMAE with rank/ordering preservation instead of distance preservation."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("mmae_lambda", 1.0)
        self.n_triplets = config.get("n_triplets", 256)
        self.use_triplet = config.get("use_triplet_rank", True)
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, ref_emb=None):
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        if ref_emb is not None:
            if self.use_triplet:
                dist_loss = rank_loss(z, ref_emb, self.n_triplets)
            else:
                dist_loss = soft_rank_correlation_loss(z, ref_emb)
            total_loss = rec_loss + self.lam * dist_loss
            return total_loss, {
                "recon_loss": rec_loss.item(),
                "rank_loss": dist_loss.item()
            }
        else:
            return rec_loss, {"recon_loss": rec_loss.item()}


# =============================================================================
# Variant 3: Local Linearity Adaptive (MMAE-Local)
# =============================================================================
# Idea: Weight distance preservation by local linearity.
# Curved regions (high PCA residual) get less weight -> allowed to unbend.
# Flat regions (low PCA residual) get high weight -> must preserve.

def estimate_local_curvature(x, k=10):
    """Estimate local curvature via PCA residual in k-neighborhood."""
    B = x.shape[0]
    device = x.device
    
    if B <= k:
        return torch.ones(B, device=device)
    
    # Pairwise distances
    dists = torch.cdist(x, x, p=2)
    
    # k-nearest neighbors (excluding self)
    _, knn_idx = dists.topk(k + 1, largest=False, dim=1)
    knn_idx = knn_idx[:, 1:]  # exclude self
    
    curvatures = []
    for i in range(B):
        neighbors = x[knn_idx[i]]  # (k, d)
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        
        # PCA via SVD
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            # Curvature proxy: ratio of smaller to larger singular values
            # High ratio = flat, low ratio = curved
            linearity = S[0] / (S.sum() + 1e-8)
            curvatures.append(1 - linearity)  # invert so high = curved
        except:
            curvatures.append(torch.tensor(0.5, device=device))
    
    return torch.stack(curvatures)


def curvature_weighted_distance_loss(z, ref_emb, x_original, k=10):
    """Distance preservation weighted by inverse local curvature."""
    B = z.shape[0]
    device = z.device
    
    # Estimate curvature from original data
    curvature = estimate_local_curvature(x_original, k=k)  # (B,)
    
    # Distance matrices
    z_dist = torch.cdist(z, z, p=2)
    ref_dist = torch.cdist(ref_emb, ref_emb, p=2)
    
    # Weight matrix: low weight for high-curvature pairs
    # w_ij = 1 - max(curv_i, curv_j)
    curv_i = curvature.unsqueeze(1).expand(B, B)
    curv_j = curvature.unsqueeze(0).expand(B, B)
    weights = 1 - torch.max(curv_i, curv_j)
    weights = torch.clamp(weights, min=0.1)  # minimum weight
    
    # Normalize distances
    z_dist_norm = z_dist / (z_dist.max() + 1e-8)
    ref_dist_norm = ref_dist / (ref_dist.max() + 1e-8)
    
    # Weighted MSE
    mask = torch.triu(torch.ones_like(z_dist), diagonal=1).bool()
    errors = (z_dist_norm - ref_dist_norm) ** 2
    weighted_errors = errors * weights
    
    return weighted_errors[mask].mean()


@register_model('mmae_local')
class MMAELocal(nn.Module):
    """MMAE with curvature-adaptive distance preservation."""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.lam = config.get("mmae_lambda", 1.0)
        self.k_neighbors = config.get("k_neighbors", 10)
        self.recon_loss = nn.MSELoss()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, ref_emb=None):
        z = self.encode(x)
        x_rec = self.decode(z)
        
        rec_loss = self.recon_loss(x_rec, x)
        
        if ref_emb is not None:
            x_flat = x.view(x.shape[0], -1)
            dist_loss = curvature_weighted_distance_loss(
                z, ref_emb, x_flat, k=self.k_neighbors
            )
            total_loss = rec_loss + self.lam * dist_loss
            return total_loss, {
                "recon_loss": rec_loss.item(),
                "dist_loss": dist_loss.item()
            }
        else:
            return rec_loss, {"recon_loss": rec_loss.item()}