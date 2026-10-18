"""
QECRL — Dynamic Detector Error Model Graph Updater (Upgraded v2)
================================================================
Implements:
  1. Exact Stim DEM Sensitivity Projection Matrix (Pi)
  2. Change-Point Adaptive CUSUM-EMA Syndrome Tracker (eliminates W=250 dropouts)
  3. Sparse Non-Negative Syndrome Deconvolution (eliminates sidelobe leakage)
  4. Calibrated Log-Odds Bayesian Weight Shift + RL Residual Modulation
  5. Incident-Edge Max-Pooling (prevents multi-qubit double counting)
  6. Self-Supervised Label-Free Hardware Proxy Reward (for live QPU deployment)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import nnls
import pymatching
import stim

from ..utils.stim_circuits import get_surface_code_layout, build_per_qubit_surface_code


class DynamicDEMUpdater:
    """
    Bridges RL continuous action space and PyMatching C++ matching graph
    via sparse spatio-temporal syndrome deconvolution.
    """

    def __init__(
        self,
        distance: int,
        nominal_circuit: stim.Circuit,
        beta: float = 3.6,
        w_min: float = 2.85,
        w_max: float = 14.0,
        score_threshold: Optional[float] = None,
        alpha_threshold: Optional[float] = None,
        p0: float = 0.001,
        p_cal: float = 0.040,
    ):
        self.distance = distance
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max
        self.p0 = p0
        self.p_cal = p_cal
        self.rounds = distance

        if alpha_threshold is None or alpha_threshold == 0.20:
            self.alpha_threshold = 0.12 if distance <= 3 else 0.18
        else:
            self.alpha_threshold = alpha_threshold

        # Distance-aware default threshold (used as detection gate)
        if score_threshold is None:
            self.score_threshold = 0.10 if distance <= 3 else 0.15
        else:
            self.score_threshold = score_threshold

        # 1. Layout and physical qubit categorization
        self.active_qubits, self.qubit_coords, self.data_qubits, self.ancilla_qubits = (
            get_surface_code_layout(distance)
        )
        self.num_qubits = len(self.active_qubits)
        self.q2idx = {q: i for i, q in enumerate(self.active_qubits)}
        self.data_mask = np.array(
            [1.0 if q in self.data_qubits else 0.0 for q in self.active_qubits],
            dtype=np.float64,
        )

        # 2. Extract nominal DEM and baseline Matching graph
        self.dem = nominal_circuit.detector_error_model()
        self.nominal_matcher = pymatching.Matching.from_detector_error_model(self.dem)
        self.det_coords = nominal_circuit.get_detector_coordinates()
        self.raw_edges = self.nominal_matcher.edges()
        self.num_edges = len(self.raw_edges)
        self.num_detectors = self.dem.num_detectors
        self.nominal_weights = np.array(
            [data["weight"] for _, _, data in self.raw_edges], dtype=np.float64
        )

        # 3. Build Exact Stim DEM Sensitivity Matrix Pi
        self.projection_matrix = self._build_exact_sensitivity_matrix()

        # 4. Precompute Calibrated Spatial Signatures for Deconvolution
        self.r0, self.diff_signatures, self.sig_norms, self.sig_norm = (
            self._build_spatial_signatures()
        )
        self.D_data = (self.diff_signatures * self.data_mask[:, np.newaxis]).T

        # 5. Change-Point Adaptive EMA Temporal State
        self.ema_rates = self.r0.copy()
        self.ema_initialized = False

    def _build_exact_sensitivity_matrix(self) -> np.ndarray:
        """
        Precompute exact continuous DEM edge sensitivity matrix Pi in [0, 1]^(num_edges x num_qubits)
        by probing Stim's detector error model compiler per physical qubit.
        """
        pi = np.zeros((self.num_edges, self.num_qubits), dtype=np.float64)
        w_ref = max(1e-6, np.log((1.0 - self.p0) / self.p0) - np.log((1.0 - self.p_cal) / self.p_cal))

        for q_idx in range(self.num_qubits):
            p_probe = np.full(self.num_qubits, self.p0, dtype=np.float64)
            p_probe[q_idx] = self.p_cal
            c_probe = build_per_qubit_surface_code(self.distance, self.rounds, p_probe)
            m_probe = pymatching.Matching.from_detector_error_model(
                c_probe.detector_error_model()
            )
            for e_idx, (u, v, data) in enumerate(self.raw_edges):
                w_nom = data["weight"]
                if v is None:
                    w_probe = m_probe.get_boundary_edge_data(u)["weight"] if m_probe.has_boundary_edge(u) else w_nom
                else:
                    w_probe = m_probe.get_edge_data(u, v)["weight"] if m_probe.has_edge(u, v) else w_nom
                delta_w = max(0.0, w_nom - w_probe)
                if delta_w > 0.01:
                    pi[e_idx, q_idx] = delta_w / w_ref

        return pi

    def _build_spatial_signatures(
        self, n_cal_shots: int = 2500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute baseline detector firing vector r0 and per-qubit differential
        firing signatures D_sig at p_cal = 0.040.
        """
        rng = np.random.default_rng(12345)
        c_base = build_per_qubit_surface_code(
            self.distance, self.rounds, np.full(self.num_qubits, self.p0)
        )
        s0 = c_base.compile_detector_sampler(seed=int(rng.integers(1, 100000))).sample(
            shots=n_cal_shots, separate_observables=True
        )[0]
        r0 = np.mean(s0, axis=0).astype(np.float64)

        signatures = np.zeros((self.num_qubits, self.num_detectors), dtype=np.float64)
        for q_idx in range(self.num_qubits):
            p_probe = np.full(self.num_qubits, self.p0, dtype=np.float64)
            p_probe[q_idx] = self.p_cal
            c_probe = build_per_qubit_surface_code(self.distance, self.rounds, p_probe)
            s_q = c_probe.compile_detector_sampler(seed=int(rng.integers(1, 100000))).sample(
                shots=n_cal_shots, separate_observables=True
            )[0]
            signatures[q_idx] = np.mean(s_q, axis=0)

        diff_sig = np.maximum(0.0, signatures - r0)
        norms = np.linalg.norm(diff_sig, axis=1)
        sig_norm = diff_sig / np.maximum(norms[:, np.newaxis], 1e-6)
        return r0, diff_sig, norms, sig_norm

    def _update_temporal_filter(self, raw_rates: np.ndarray) -> np.ndarray:
        """
        Change-Point Adaptive CUSUM-EMA Filter:
          - If sudden onset or turn-off jump is detected, fast-track (alpha = 0.78 - 0.85)
          - During steady plateau, smooth out W=250 binomial variance (alpha = 0.38)
        """
        if not self.ema_initialized:
            self.ema_rates = raw_rates.copy()
            self.ema_initialized = True
            return self.ema_rates

        # Vector infinity-norm change-point detection across all detectors
        jump_norm = float(np.max(np.abs(raw_rates - self.ema_rates)))
        max_inst_excess = float(np.max(raw_rates - self.r0))
        max_ema_excess = float(np.max(self.ema_rates - self.r0))

        if max_inst_excess < 0.012 and max_ema_excess > 0.020:
            # Defect turned OFF -> immediate reset to avoid stale ghost weights
            self.ema_rates = 0.85 * raw_rates + 0.15 * self.r0
        elif jump_norm > 0.015:
            # Abrupt local or global jump -> fast acquisition
            self.ema_rates = 0.80 * raw_rates + 0.20 * self.ema_rates
        else:
            # Steady plateau -> variance-suppressing smoothing
            self.ema_rates = 0.38 * raw_rates + 0.62 * self.ema_rates

        return self.ema_rates

    def compute_matched_filter_scores(self, detector_firing_rates: np.ndarray) -> np.ndarray:
        """Compute raw matched-filter projection scores."""
        obs_diff = np.maximum(0.0, detector_firing_rates - self.r0)
        return (self.sig_norm @ obs_diff).astype(np.float64)

    def deconvolve_defect_probabilities(
        self, detector_firing_rates: np.ndarray, advance_filter: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sparse Syndrome Deconvolution via Non-Negative Least Squares (NNLS):
        Jointly solves:
          min_{alpha >= 0} || D_data * alpha - residual ||_2^2
        eliminating spatial sidelobe leakage onto neighboring qubits and ancillae.

        Returns:
            p_hat: Estimated per-qubit error rate vector of shape (num_qubits,)
            alpha_est: Defect amplitude vector in [0, 1.35]
            scores: Projection scores of shape (num_qubits,)
        """
        if advance_filter:
            filtered_rates = self._update_temporal_filter(detector_firing_rates)
        else:
            filtered_rates = self.ema_rates if self.ema_initialized else detector_firing_rates
        residual = np.maximum(0.0, filtered_rates - self.r0)

        # High-performance non-negative deconvolution on data qubits
        alpha_raw, _ = nnls(self.D_data, residual)
        alpha_est = np.zeros(self.num_qubits, dtype=np.float64)
        alpha_est[self.data_mask > 0.5] = np.clip(alpha_raw[self.data_mask > 0.5], 0.0, 1.35)

        p_hat = self.p0 + alpha_est * (self.p_cal - self.p0)
        p_hat = np.clip(p_hat, self.p0, 0.065)

        # Matched filter scores on filtered residual
        scores = (self.sig_norm @ residual) * self.data_mask
        return p_hat, alpha_est, scores

    def update_graph(
        self,
        action: np.ndarray,
        detector_firing_rates: Optional[np.ndarray] = None,
        advance_filter: bool = True,
        **kwargs,
    ) -> pymatching.Matching:
        """
        Reweight PyMatching graph using sparse-deconvolved Bayesian log-odds
        priors modulated by the RL agent's continuous action vector.
        """
        action = np.asarray(action, dtype=np.float64)

        if detector_firing_rates is not None:
            p_hat, alpha_est, scores = self.deconvolve_defect_probabilities(
                detector_firing_rates, advance_filter=advance_filter
            )
            active_gate = (alpha_est > self.alpha_threshold) & (self.data_mask > 0.5)

            # Exact Bayesian log-odds shift per defective qubit:
            # Delta_w_q = log((1 - p0)/p0) - log((1 - p_hat)/p_hat)
            w_nom_q = np.log((1.0 - self.p0) / self.p0)
            w_hat_q = np.log((1.0 - p_hat) / p_hat)
            bayes_delta_q = np.maximum(0.0, w_nom_q - w_hat_q)

            # RL action in [-1, 1] acts as fine-grained residual gain around Bayesian prior
            # Maps action in [-1, 1] -> multiplier in [0.60, 1.45] when active; exactly 1.0 at action=0
            rl_gain = np.clip(1.0 + 0.40 * action, 0.60, 1.45)
            qubit_weight_shifts = np.where(active_gate, bayes_delta_q * rl_gain, 0.0)

            # Continuous sensitivity projection bounded by max individual shift
            max_shift = float(np.max(qubit_weight_shifts)) if np.any(active_gate) else 0.0
            edge_shifts = np.minimum(self.projection_matrix @ qubit_weight_shifts, max_shift)
            updated_weights = np.clip(
                self.nominal_weights - edge_shifts, self.w_min, self.w_max
            )
        else:
            # Fallback if called without telemetry rates
            edge_act = np.clip(self.projection_matrix @ np.maximum(0.0, action), 0.0, 1.0)
            delta_w = self.beta * edge_act
            updated_weights = np.clip(self.nominal_weights - delta_w, self.w_min, self.w_max)

        reweighted_matcher = pymatching.Matching()
        for idx, (u, v, data) in enumerate(self.raw_edges):
            w = float(updated_weights[idx])
            f = data["fault_ids"]
            if v is None:
                reweighted_matcher.add_boundary_edge(u, weight=w, fault_ids=f)
            else:
                reweighted_matcher.add_edge(u, v, weight=w, fault_ids=f)

        return reweighted_matcher

    def compute_self_supervised_reward(
        self, detector_firing_rates: np.ndarray, action: np.ndarray
    ) -> float:
        """
        Hardware-Native Label-Free Proxy Reward:
        Measures how accurately the RL-modulated DEM error model explains the
        observed syndrome firing excess without requiring ground-truth logical flips.
        """
        obs_excess = np.maximum(0.0, detector_firing_rates - self.r0)
        p_hat, alpha_est, scores = self.deconvolve_defect_probabilities(detector_firing_rates)
        active_gate = (alpha_est > self.alpha_threshold) & (self.data_mask > 0.5)

        if not np.any(active_gate):
            return 0.0

        rl_gain = np.clip(1.0 + 0.40 * np.asarray(action, dtype=np.float64), 0.60, 1.45)
        alpha_vec = np.where(
            active_gate,
            ((p_hat - self.p0) / (self.p_cal - self.p0)) * rl_gain,
            0.0,
        )
        reconstructed_excess = alpha_vec @ self.diff_signatures
        unexplained_error = np.linalg.norm(obs_excess - reconstructed_excess)
        baseline_error = np.linalg.norm(obs_excess) + 1e-6

        # Positive reward in [0, 10] proportional to explained syndrome variance
        explained_ratio = 1.0 - (unexplained_error / baseline_error)
        return float(np.clip(explained_ratio * 10.0, -5.0, 10.0))

    def reset(self):
        """Reset temporal EMA tracker between independent episodes/runs."""
        self.ema_rates = self.r0.copy()
        self.ema_initialized = False
