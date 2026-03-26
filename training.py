"""Training utilities."""

import torch
import numpy as np
from collections import defaultdict


class Trainer:
    def __init__(self, model, optimizer, device='cuda', model_name=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.history = defaultdict(list)
    
    def train_epoch(self, loader):
        self.model.train()
        losses = defaultdict(list)
        
        for batch in loader:
            # Handle different batch formats:
            # - (x, y): vanilla, topoae, rtdae, geomae
            # - (x, emb, y): mmae
            # - (x, idx, y): ggae
            # - (x, emb, idx, y): ggae with embeddings (unused)
            
            if self.model_name == 'ggae':
                if len(batch) == 4:  # (x, emb, idx, y)
                    x, emb, indices, _ = batch
                    x = x.to(self.device)
                    indices = indices.to(self.device)
                else:  # (x, idx, y)
                    x, indices, _ = batch
                    x = x.to(self.device)
                    indices = indices.to(self.device)
                emb = None
            elif len(batch) == 3:  # (x, emb, y) for MMAE
                x, emb, _ = batch
                x, emb = x.to(self.device), emb.to(self.device)
                indices = None
            else:  # (x, y)
                x, _ = batch
                x = x.to(self.device)
                emb = None
                indices = None
            
            self.optimizer.zero_grad()
            
            # Call model with appropriate arguments
            if self.model_name == 'mmae' and emb is not None:
                loss, comps = self.model(x, emb)
            elif self.model_name == 'ggae':
                loss, comps = self.model(x, indices)
            else:
                loss, comps = self.model(x)
            
            loss.backward()
            self.optimizer.step()
            
            losses['total_loss'].append(loss.item())
            for k, v in comps.items():
                losses[k].append(v)
        
        return {k: np.mean(v) for k, v in losses.items()}
    
    def evaluate(self, loader):
        self.model.eval()
        losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in loader:
                # Same batch handling as train_epoch
                if self.model_name == 'ggae':
                    if len(batch) == 4:
                        x, emb, indices, _ = batch
                        x = x.to(self.device)
                        indices = indices.to(self.device)
                    else:
                        x, indices, _ = batch
                        x = x.to(self.device)
                        indices = indices.to(self.device)
                    emb = None
                elif len(batch) == 3:
                    x, emb, _ = batch
                    x, emb = x.to(self.device), emb.to(self.device)
                    indices = None
                else:
                    x, _ = batch
                    x = x.to(self.device)
                    emb = None
                    indices = None
                
                if self.model_name == 'mmae' and emb is not None:
                    loss, comps = self.model(x, emb)
                elif self.model_name == 'ggae':
                    loss, comps = self.model(x, indices)
                else:
                    loss, comps = self.model(x)
                
                losses['total_loss'].append(loss.item())
                for k, v in comps.items():
                    losses[k].append(v)
        
        return {k: np.mean(v) for k, v in losses.items()}
    
    def fit(self, train_loader, val_loader=None, n_epochs=100, verbose=True):
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            for k, v in train_loss.items():
                self.history[f'train_{k}'].append(v)
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                for k, v in val_loss.items():
                    self.history[f'val_{k}'].append(v)
            
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{n_epochs} - loss: {train_loss['total_loss']:.4f}"
                if 'gg_loss' in train_loss:
                    msg += f" - gg_loss: {train_loss['gg_loss']:.4f}"
                if val_loader:
                    msg += f" - val_loss: {val_loss['total_loss']:.4f}"
                print(msg)
        
        return self.history


def get_latents(model, loader, device='cuda'):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for batch in loader:
            # Handle all batch formats
            if len(batch) == 4:  # (x, emb, idx, y)
                x, _, _, y = batch
            elif len(batch) == 3:  # (x, emb, y) or (x, idx, y)
                x, _, y = batch
            else:  # (x, y)
                x, y = batch
            latents.append(model.encode(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(latents), np.concatenate(labels)


def get_reconstructions(model, loader, device='cuda'):
    """Get original data and reconstructions for evaluation."""
    model.eval()
    originals, reconstructions, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            # Handle all batch formats
            if len(batch) == 4:  # (x, emb, idx, y)
                x, _, _, y = batch
            elif len(batch) == 3:  # (x, emb, y) or (x, idx, y)
                x, _, y = batch
            else:  # (x, y)
                x, y = batch
            x = x.to(device)
            z = model.encode(x)
            x_rec = model.decode(z)
            originals.append(x.cpu().numpy())
            reconstructions.append(x_rec.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(originals), np.concatenate(reconstructions), np.concatenate(labels)