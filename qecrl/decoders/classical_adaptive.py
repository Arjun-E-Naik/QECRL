"""
QECRL — Classical Adaptive Baselines
=====================================
Implements non-ML adaptive decoders for benchmarking against RL-MWPM:

  1. Spitz-Corr-MWPM:
     Analytical pairwise syndrome correlation inversion based on Spitz et al.
     Inverts empirical co-firing probabilities into edge error rates:
       p_e_hat = (1/W) * sum_k (D_{k,u} ^ D_{k,v})
       w_e_hat = log((1 - p_e_hat) / p_e_hat)

  2. EMA-Heuristic-MWPM:
     Smooth Exponential Moving Average (EMA) detector firing heuristic.
     Scales nominal edge weights inversely with adjacent detector activity:
       w_e(t) = clip(w_e_nominal - beta * log(mu_local / mu_0), w_min, w_max)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pymatching
import stim


class SpitzCorrMWPM:
    """
    Classical Adaptive Baseline: Analytical Pairwise Correlation Inversion.
    Uses Spitz et al. formula over sliding window of W shots.
    """

    def __init__(
        self,
        nominal_circuit: stim.Circuit,
        p_min: float = 1e-4,
        p_max: float = 0.25,
        w_min: float = 1.0,
        w_max: float = 14.0,
    ):
        self.p_min = p_min
        self.p_max = p_max
        self.w_min = w_min
        self.w_max = w_max

        self.dem = nominal_circuit.detector_error_model()
        self.base_matcher = pymatching.Matching.from_detector_error_model(self.dem)
        self.edges = self.base_matcher.edges()

    def decode_window(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Invert correlations from window of W shots, update weights, and decode.

        Args:
            syndromes: Array of shape (W, num_detectors)

        Returns:
            Decoded observable predictions of shape (W, num_observables)
        """
        W, N = syndromes.shape
        matcher = self.reweight(syndromes)
        return matcher.decode_batch(syndromes)

    def reweight(self, syndromes: np.ndarray) -> pymatching.Matching:
        """Compute analytical weights and return reweighted Matching graph."""
        W, N = syndromes.shape
        matcher = pymatching.Matching()

        for u, v, data in self.edges:
            f = data["fault_ids"]
            if v is None:
                # Boundary edge: error probability estimated by single detector firing rate
                p_hat = np.mean(syndromes[:, u])
            else:
                # Internal edge: error probability estimated by parity co-firing
                diff = np.bitwise_xor(syndromes[:, u], syndromes[:, v])
                p_hat = np.mean(diff)

            p_hat = float(np.clip(p_hat, self.p_min, self.p_max))
            w = float(np.clip(np.log((1.0 - p_hat) / p_hat), self.w_min, self.w_max))

            if v is None:
                matcher.add_boundary_edge(u, weight=w, fault_ids=f)
            else:
                matcher.add_edge(u, v, weight=w, fault_ids=f)

        return matcher


class EMAHeuristicMWPM:
    """
    Classical Heuristic Baseline: Exponential Moving Average Firing Rate.
    Smoothly tracks detector activity and adjusts weights proportionally.
    """

    def __init__(
        self,
        nominal_circuit: stim.Circuit,
        nominal_p: float = 0.005,
        alpha: float = 0.15,
        beta: float = 2.0,
        w_min: float = 1.0,
        w_max: float = 14.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max

        self.dem = nominal_circuit.detector_error_model()
        self.base_matcher = pymatching.Matching.from_detector_error_model(self.dem)
        self.edges = self.base_matcher.edges()
        self.num_detectors = self.base_matcher.num_detectors

        # Expected nominal detector firing probability
        self.mu_0 = max(nominal_p * 2.0, 1e-4)
        self.ema_rates = np.full(self.num_detectors, self.mu_0, dtype=np.float64)

    def decode_window(self, syndromes: np.ndarray) -> np.ndarray:
        matcher = self.reweight(syndromes)
        return matcher.decode_batch(syndromes)

    def reweight(self, syndromes: np.ndarray) -> pymatching.Matching:
        """Update EMA rates and rebuild graph with adjusted weights."""
        current_rates = np.mean(syndromes, axis=0)
        self.ema_rates = (1.0 - self.alpha) * self.ema_rates + self.alpha * current_rates

        matcher = pymatching.Matching()
        for u, v, data in self.edges:
            w_nom = data["weight"]
            f = data["fault_ids"]

            if v is None:
                local_rate = self.ema_rates[u]
            else:
                local_rate = (self.ema_rates[u] + self.ema_rates[v]) / 2.0

            ratio = max(local_rate / self.mu_0, 0.05)
            # Higher local rate -> higher error -> reduce edge weight
            w = float(np.clip(w_nom - self.beta * np.log(ratio), self.w_min, self.w_max))

            if v is None:
                matcher.add_boundary_edge(u, weight=w, fault_ids=f)
            else:
                matcher.add_edge(u, v, weight=w, fault_ids=f)

        return matcher

    def reset(self):
        self.ema_rates.fill(self.mu_0)
