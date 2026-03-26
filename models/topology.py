"""Persistent homology computation."""

import numpy as np


class UnionFind:
    def __init__(self, n):
        self._parent = np.arange(n, dtype=int)
    
    def find(self, u):
        if self._parent[u] == u:
            return u
        self._parent[u] = self.find(self._parent[u])
        return self._parent[u]
    
    def merge(self, u, v):
        if u != v:
            self._parent[self.find(u)] = self.find(v)


class PersistentHomologyCalculation:
    """Compute 0-dim persistent homology."""
    
    def __call__(self, matrix):
        n = matrix.shape[0]
        uf = UnionFind(n)
        
        triu_idx = np.triu_indices_from(matrix)
        weights = matrix[triu_idx]
        order = np.argsort(weights, kind='stable')
        
        pairs = []
        for idx in order:
            u, v = triu_idx[0][idx], triu_idx[1][idx]
            cu, cv = uf.find(u), uf.find(v)
            if cu == cv:
                continue
            if cu > cv:
                uf.merge(v, u)
            else:
                uf.merge(u, v)
            pairs.append((min(u, v), max(u, v)))
        
        return np.array(pairs), np.array([])