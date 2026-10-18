"""
QECRL — Surface Code Reweighting Gymnasium Environment (RL-MWPM)
================================================================
Gymnasium environment for adaptive QEC graph reweighting over sliding
windows of syndrome measurement shots.

Follows the Gymnasium API and is fully compatible with Stable-Baselines3
(PPO, SAC, TD3) and contextual bandit agents.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stim
import pymatching

from ..utils.stim_circuits import get_surface_code_layout, build_per_qubit_surface_code
from ..noise.noise_engine import NonStationaryNoiseEngine
from ..features.syndrome_features import SyndromeFeatureExtractor
from ..decoders.graph_updater import DynamicDEMUpdater


class SurfaceCodeReweightEnv(gym.Env):
    """
    Gymnasium Environment for Adaptive RL-Guided Detector Error Model Reweighting.

    Observation:
      Compact statistical vector S_t in R^M containing:
        - Detector firing frequencies: <d_i>
        - DEM edge covariances: C_{ij} for (i, j) in E_DEM
        - Drift momentum: Delta S_t

    Action:
      Continuous vector a_t in [-1, 1]^(N_qubits):
        Per-physical-qubit log-probability multipliers updating DEM graph weights.

    Reward:
      Performance differential against static MWPM baseline:
        R_t = 100 * (LER_static - LER_agent) - lambda * ||a_t - a_{t-1}||^2
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        distance: int = 3,
        rounds: Optional[int] = None,
        window_size: int = 250,
        max_windows: int = 100,
        regime: str = "all",
        p0: float = 0.001,
        beta: float = 3.6,
        smoothness_penalty: float = 0.0001,
        reward_mode: str = "sim",  # 'sim', 'proxy', 'hybrid'
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.distance = distance
        self.rounds = rounds if rounds is not None else distance
        self.window_size = window_size
        self.max_windows = max_windows
        self.regime = regime
        self.p0 = p0
        self.beta = beta
        self.smoothness_penalty = smoothness_penalty
        self.reward_mode = reward_mode

        # 1. Physical layout and nominal baseline circuit
        self.active_qubits, self.qubit_coords, self.data_qubits, self.ancilla_qubits = get_surface_code_layout(distance)
        self.num_qubits = len(self.active_qubits)

        nominal_p_vec = np.full(self.num_qubits, p0, dtype=np.float64)
        self.nominal_circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.rounds, p_vec=nominal_p_vec
        )
        self.nominal_dem = self.nominal_circuit.detector_error_model()
        self.num_detectors = self.nominal_dem.num_detectors
        self.num_observables = self.nominal_dem.num_observables

        # Static MWPM baseline decoder (never updated)
        self.static_matcher = pymatching.Matching.from_detector_error_model(self.nominal_dem)

        # 2. Dynamic graph updater & feature extractor
        self.graph_updater = DynamicDEMUpdater(
            distance=distance,
            nominal_circuit=self.nominal_circuit,
            beta=beta,
        )

        dem_edges = [(u, v) for u, v, _ in self.graph_updater.raw_edges]
        self.feature_extractor = SyndromeFeatureExtractor(
            num_detectors=self.num_detectors,
            dem_edges=dem_edges,
        )

        # 3. Non-stationary noise channel
        self.noise_engine = NonStationaryNoiseEngine(
            distance=distance,
            regime=regime,
            p0=p0,
            seed=seed,
        )

        # 4. Gymnasium Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_qubits,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.feature_extractor.feature_dim,), dtype=np.float32
        )

        # Internal state
        self.current_window = 0
        self.prev_action = np.zeros(self.num_qubits, dtype=np.float32)
        self.current_syndromes: Optional[np.ndarray] = None
        self.current_observables: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to begin a new non-stationary episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.noise_engine.reset(seed=seed)
        else:
            self.noise_engine.reset()

        self.feature_extractor.reset()
        self.current_window = 0
        self.prev_action.fill(0.0)

        # Step noise once and sample initial syndrome window
        p_vec = self.noise_engine.step()
        circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.rounds, p_vec=p_vec
        )
        sampler = circuit.compile_detector_sampler()
        self.current_syndromes, self.current_observables = sampler.sample(
            shots=self.window_size, separate_observables=True
        )

        obs = self.feature_extractor.extract(self.current_syndromes)
        info = {
            "window": self.current_window,
            "mean_p": float(np.mean(p_vec)),
            "max_p": float(np.max(p_vec)),
        }
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one outer-loop reweighting step.

        Args:
            action: Continuous array of shape (num_qubits,) in [-1, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.current_window += 1

        # 1. Compute empirical detector firing rates from telemetry window
        firing_rates = np.mean(self.current_syndromes, axis=0)

        # 2. Update PyMatching graph using agent's action with threshold gating
        agent_matcher = self.graph_updater.update_graph(
            action,
            detector_firing_rates=firing_rates,
        )

        # 3. Decode evaluation batch using Static vs Agent matcher
        # Sample evaluation batch (N_eval = 2000) to eliminate Poisson shot-noise variance
        eval_circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.rounds, p_vec=self.noise_engine.get_p_vec()
        )
        eval_sampler = eval_circuit.compile_detector_sampler()
        eval_dets, eval_obs = eval_sampler.sample(shots=2000, separate_observables=True)

        static_preds = self.static_matcher.decode_batch(eval_dets)
        static_ler = float(np.mean(np.any(static_preds != eval_obs, axis=1)))

        agent_preds = agent_matcher.decode_batch(eval_dets)
        agent_ler = float(np.mean(np.any(agent_preds != eval_obs, axis=1)))

        # 4. Compute Reward: pure LER improvement with minimal action penalty
        smoothness_penalty = self.smoothness_penalty * float(np.sum((action - self.prev_action) ** 2))
        ler_delta = (static_ler - agent_ler) * 100.0  # Positive when agent beats static

        reward = ler_delta - smoothness_penalty
        self.prev_action = action.copy()

        # 4. Advance noise engine and sample next window of syndromes
        p_vec = self.noise_engine.step()
        next_circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.rounds, p_vec=p_vec
        )
        sampler = next_circuit.compile_detector_sampler()
        self.current_syndromes, self.current_observables = sampler.sample(
            shots=self.window_size, separate_observables=True
        )

        # 5. Extract next observation
        next_obs = self.feature_extractor.extract(self.current_syndromes)

        terminated = False
        truncated = self.current_window >= self.max_windows

        info = {
            "window": self.current_window,
            "static_ler": static_ler,
            "agent_ler": agent_ler,
            "ler_improvement": static_ler - agent_ler,
            "mean_p": float(np.mean(p_vec)),
            "max_p": float(np.max(p_vec)),
        }

        return next_obs, reward, terminated, truncated, info

    def render(self):
        return (
            f"[Window {self.current_window:3d}/{self.max_windows}] "
            f"Noise ⟨p⟩={np.mean(self.noise_engine.get_p_vec()):.4f}"
        )
