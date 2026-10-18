"""
QECRL — Syndrome Feature Extractor
===================================
Extracts compact, physically informed statistical features from a sliding
window of syndrome measurement shots:
  1. Detector firing frequencies: <d_i> = (1/W) sum_k D_{k,i}
  2. Local edge covariances: C_{ij} = <d_i d_j> - <d_i><d_j> for (i, j) in E_DEM
  3. Drift momentum: Exponential moving average difference Delta S_t
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


class SyndromeFeatureExtractor:
    """
    Computes first-order and second-order syndrome statistics over a window
    of W shots, restricted to the edges present in the Detector Error Model.
    """

    def __init__(
        self,
        num_detectors: int,
        dem_edges: List[Tuple[int, Optional[int]]],
        momentum_decay: float = 0.8,
    ):
        self.num_detectors = num_detectors
        # Filter only two-detector pairs for second-order correlation
        self.internal_edges = [(u, v) for (u, v) in dem_edges if v is not None]
        self.edge_u = np.array([u for u, v in self.internal_edges], dtype=np.int32)
        self.edge_v = np.array([v for u, v in self.internal_edges], dtype=np.int32)
        self.num_edges = len(self.internal_edges)

        self.momentum_decay = momentum_decay
        self.prev_firing_rates = np.zeros(num_detectors, dtype=np.float32)
        self.initialized = False

        # State dimensionality: num_detectors (rates) + num_edges (cov) + num_detectors (momentum)
        self.feature_dim = 2 * num_detectors + self.num_edges

    def extract(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from binary syndrome matrix.

        Args:
            syndromes: Array of shape (W, num_detectors) with dtype uint8 or bool

        Returns:
            State vector S_t of shape (feature_dim,) with dtype float32
        """
        W, N = syndromes.shape
        if N != self.num_detectors:
            raise ValueError(f"Expected {self.num_detectors} detectors, got {N}")

        syndromes_f = syndromes.astype(np.float32)

        # 1. First-order detector firing rates: <d_i>
        firing_rates = np.mean(syndromes_f, axis=0)  # Shape (num_detectors,)

        # 2. Second-order edge covariances for (u, v) in E_DEM
        if self.num_edges > 0:
            # Vectorized dot product for all DEM edges: (1/W) * sum_k D[k, u] * D[k, v]
            du = syndromes_f[:, self.edge_u]  # (W, num_edges)
            dv = syndromes_f[:, self.edge_v]  # (W, num_edges)
            co_firing = np.mean(du * dv, axis=0)  # (num_edges,)
            independent_prob = firing_rates[self.edge_u] * firing_rates[self.edge_v]
            edge_covariances = co_firing - independent_prob
        else:
            edge_covariances = np.empty(0, dtype=np.float32)

        # 3. Drift momentum: Delta S_t = firing_rates - prev_firing_rates
        if not self.initialized:
            momentum = np.zeros_like(firing_rates)
            self.prev_firing_rates = firing_rates.copy()
            self.initialized = True
        else:
            momentum = firing_rates - self.prev_firing_rates
            self.prev_firing_rates = (
                self.momentum_decay * self.prev_firing_rates + (1 - self.momentum_decay) * firing_rates
            )

        features = np.concatenate([firing_rates, edge_covariances, momentum])
        return features.astype(np.float32)

    def reset(self):
        """Reset internal momentum state."""
        self.prev_firing_rates.fill(0.0)
        self.initialized = False
