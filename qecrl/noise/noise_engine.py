"""
QECRL — Non-Stationary Noise Physics Engine
============================================
Simulates realistic, heterogeneous, time-varying noise channels on rotated
surface codes across three hardware fault regimes:

  Regime A: Continuous Thermal Drift (Sinusoidal + Ornstein-Uhlenbeck random walk)
  Regime B: Two-Level System (TLS) Telegraph Noise (Markovian hot-defect switching)
  Regime C: Spatial Burst / Crosstalk Events (Localized cosmic-ray / phonon clusters)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..utils.stim_circuits import get_surface_code_layout


class ThermalDriftGenerator:
    """
    Regime A: Continuous Thermal Drift
    Each physical qubit q drifts continuously via:
      p_q(t) = clip( p_0 * [1 + alpha_q * sin(2*pi*t / tau_q + phi_q) + epsilon_q(t)], p_min, p_max )
    where epsilon_q(t) follows an Ornstein-Uhlenbeck mean-reverting process.
    """

    def __init__(
        self,
        num_qubits: int,
        p0: float = 0.001,
        p_min: float = 0.0002,
        p_max: float = 0.005,
        ou_theta: float = 0.05,
        ou_sigma: float = 0.02,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_qubits = num_qubits
        self.p0 = p0
        self.p_min = p_min
        self.p_max = p_max
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.rng = rng if rng is not None else np.random.default_rng()

        self.alpha = self.rng.uniform(0.15, 0.40, size=num_qubits)
        self.tau = self.rng.uniform(40.0, 120.0, size=num_qubits)
        self.phi = self.rng.uniform(0.0, 2 * np.pi, size=num_qubits)
        self.epsilon = np.zeros(num_qubits, dtype=np.float64)
        self.t = 0

    def step(self) -> np.ndarray:
        noise = self.rng.normal(0.0, 1.0, size=self.num_qubits)
        self.epsilon += -self.ou_theta * self.epsilon + self.ou_sigma * noise

        sin_mod = self.alpha * np.sin(2 * np.pi * self.t / self.tau + self.phi)
        p_vec = self.p0 * (1.0 + sin_mod + self.epsilon)
        self.t += 1
        return np.clip(p_vec, self.p_min, self.p_max)

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self.alpha = self.rng.uniform(0.15, 0.40, size=self.num_qubits)
        self.tau = self.rng.uniform(40.0, 120.0, size=self.num_qubits)
        self.phi = self.rng.uniform(0.0, 2 * np.pi, size=self.num_qubits)
        self.epsilon = np.zeros(self.num_qubits, dtype=np.float64)
        self.t = 0


class TLSTelegraphGenerator:
    """
    Regime B: High-Contrast Two-Level System (TLS) Telegraph Noise
    Physical qubits or correlated defect clusters transition between baseline state
    p_low = 0.001 and defective 'hot' state p_high = 0.040 (40x surge) via a two-state
    Markov process. Hot defect clusters persist for sustained plateaus
    (gamma_off ~ 0.04 -> mean duration ~25 windows).
    """

    def __init__(
        self,
        num_qubits: int,
        p_low: float = 0.001,
        p_high: float = 0.040,
        gamma_on: float = 0.018,
        gamma_off: float = 0.040,
        defect_clusters: Optional[List[List[int]]] = None,
        susceptible_qubits: Optional[List[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_qubits = num_qubits
        self.p_low = p_low
        self.p_high = p_high
        self.gamma_on = gamma_on
        self.gamma_off = gamma_off
        self.rng = rng if rng is not None else np.random.default_rng()

        if defect_clusters is not None:
            self.clusters = defect_clusters
        elif susceptible_qubits is not None:
            self.clusters = [[q] for q in susceptible_qubits]
        else:
            self.clusters = [[q] for q in range(num_qubits)]

        self.num_clusters = len(self.clusters)
        self.states = np.zeros(self.num_clusters, dtype=np.int32)

    def step(self) -> np.ndarray:
        p_vec = np.full(self.num_qubits, self.p_low, dtype=np.float64)
        for i, cluster in enumerate(self.clusters):
            state = self.states[i]
            r = self.rng.uniform(0.0, 1.0)
            if state == 0 and r < self.gamma_on:
                self.states[i] = 1
            elif state == 1 and r < self.gamma_off:
                self.states[i] = 0

            if self.states[i] == 1:
                for q in cluster:
                    p_vec[q] = max(p_vec[q], self.p_high)
        return p_vec

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self.states = np.zeros(self.num_clusters, dtype=np.int32)


class SpatialBurstGenerator:
    """
    Regime C: Spatially Correlated Burst Events
    Models cosmic-ray / phonon impacts causing transient 25x error surge across
    adjacent localized qubits for 6-12 windows.
    """

    def __init__(
        self,
        active_qubits: List[int],
        coords: Dict[int, Tuple[float, float]],
        burst_prob: float = 0.04,
        surge_factor: float = 25.0,
        min_duration: int = 6,
        max_duration: int = 12,
        rng: Optional[np.random.Generator] = None,
    ):
        self.active_qubits = active_qubits
        self.coords = coords
        self.burst_prob = burst_prob
        self.surge_factor = surge_factor
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng = rng if rng is not None else np.random.default_rng()

        self.num_qubits = len(active_qubits)
        self.active_burst: Optional[Dict] = None

    def step(self) -> np.ndarray:
        multipliers = np.ones(self.num_qubits, dtype=np.float64)

        if self.active_burst is not None:
            self.active_burst["remaining_windows"] -= 1
            if self.active_burst["remaining_windows"] <= 0:
                self.active_burst = None
            else:
                for idx in self.active_burst["affected_indices"]:
                    multipliers[idx] = self.surge_factor

        if self.active_burst is None and self.rng.uniform(0.0, 1.0) < self.burst_prob:
            duration = int(self.rng.integers(self.min_duration, self.max_duration + 1))
            center_idx = int(self.rng.integers(0, self.num_qubits))
            center_q = self.active_qubits[center_idx]
            cx, cy = self.coords[center_q]

            affected = []
            for i, q in enumerate(self.active_qubits):
                qx, qy = self.coords[q]
                if abs(qx - cx) <= 2.2 and abs(qy - cy) <= 2.2:
                    affected.append(i)
                    multipliers[i] = self.surge_factor

            self.active_burst = {
                "remaining_windows": duration,
                "center": (cx, cy),
                "affected_indices": affected,
            }

        return multipliers

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self.active_burst = None


class NonStationaryNoiseEngine:
    """
    Unified Non-Stationary Noise Engine.
    High-contrast configuration with baseline p0 = 0.001 and severe localized defects.
    """

    def __init__(
        self,
        distance: int = 3,
        regime: str = "all",  # 'all', 'thermal', 'tls', 'burst'
        p0: float = 0.001,
        p_min: float = 0.0002,
        p_max: float = 0.050,
        seed: Optional[int] = None,
    ):
        self.distance = distance
        self.regime = regime.lower()
        self.p0 = p0
        self.p_min = p_min
        self.p_max = p_max
        self.rng = np.random.default_rng(seed)

        self.active_qubits, self.coords, self.data_qubits, self.ancilla_qubits = get_surface_code_layout(distance)
        self.num_qubits = len(self.active_qubits)
        q2idx = {q: i for i, q in enumerate(self.active_qubits)}

        # Construct boundary defect clusters along the logical error chain boundaries (length k = ceil(d/2))
        k = (distance + 1) // 2
        all_x = [self.coords[q][0] for q in self.data_qubits]
        min_x, max_x = min(all_x), max(all_x)

        line1 = sorted([q for q in self.data_qubits if self.coords[q][0] == min_x], key=lambda q: self.coords[q][1])
        line2 = sorted([q for q in self.data_qubits if self.coords[q][0] == max_x], key=lambda q: self.coords[q][1])

        boundary_clusters = []
        for line in [line1, line2]:
            for i in range(len(line) - k + 1):
                cluster = [q2idx[q] for q in line[i:i + k]]
                if cluster not in boundary_clusters:
                    boundary_clusters.append(cluster)

        self.thermal_gen = ThermalDriftGenerator(self.num_qubits, p0=p0, p_min=p_min, p_max=0.003, rng=self.rng)
        self.tls_gen = TLSTelegraphGenerator(
            self.num_qubits,
            p_low=p0,
            p_high=0.040,
            gamma_on=0.035,
            gamma_off=0.040,
            defect_clusters=boundary_clusters,
            rng=self.rng,
        )
        self.burst_gen = SpatialBurstGenerator(self.active_qubits, self.coords, burst_prob=0.04, surge_factor=25.0, rng=self.rng)

        self.current_p_vec = np.full(self.num_qubits, p0, dtype=np.float64)
        self.step_idx = 0
        self.history: List[np.ndarray] = []

    def step(self) -> np.ndarray:
        if self.regime == "thermal":
            p = self.thermal_gen.step()
        elif self.regime == "tls":
            p = self.tls_gen.step()
        elif self.regime == "burst":
            mult = self.burst_gen.step()
            p = np.clip(self.p0 * mult, self.p_min, self.p_max)
        elif self.regime == "all":
            p_therm = self.thermal_gen.step()
            tls_vec = self.tls_gen.step()
            p_base = np.maximum(p_therm, tls_vec)
            mult = self.burst_gen.step()
            p = np.clip(p_base * mult, self.p_min, self.p_max)
        else:
            p = np.full(self.num_qubits, self.p0, dtype=np.float64)

        self.current_p_vec = p
        self.history.append(p.copy())
        self.step_idx += 1
        return self.current_p_vec

    def get_p_vec(self) -> np.ndarray:
        return self.current_p_vec

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.thermal_gen.reset(rng=self.rng)
        self.tls_gen.reset(rng=self.rng)
        self.burst_gen.reset(rng=self.rng)
        self.current_p_vec = np.full(self.num_qubits, self.p0, dtype=np.float64)
        self.step_idx = 0
        self.history.clear()
