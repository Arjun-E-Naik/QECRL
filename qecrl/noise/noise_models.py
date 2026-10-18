"""
QECRL — Quantum Error Correction via Reinforcement Learning
===========================================================
Noise models for QEC environments.

Key insight: The research novelty of QECRL is the DRIFT.
Static noise → any decoder works fine.
Drifting noise → static decoders fail → RL decoder adapts.

Noise model hierarchy:
  DepolarizingNoise  → baseline i.i.d. depolarizing errors
  BiasedNoise        → biased toward Z (phase-flip) or X (bit-flip) errors
  CircuitLevelNoise  → includes measurement errors (realistic)
  DriftingNoise      → wraps any noise model and adds temporal drift
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import stim


# ────────────────────────────────────────────────────────────────────────────
# Base noise model
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseParams:
    """Container for current noise parameters. Updated every step during drift."""
    p: float         # Physical error probability per gate/qubit
    p_meas: float    # Measurement error probability (can differ from gate errors)
    bias_eta: float  # Bias ratio Z:X (1.0 = unbiased depolarizing)

    def __post_init__(self):
        assert 0 < self.p < 1,      f"p must be in (0,1), got {self.p}"
        assert 0 < self.p_meas < 1, f"p_meas must be in (0,1), got {self.p_meas}"
        assert self.bias_eta >= 1,  f"bias_eta must be >= 1, got {self.bias_eta}"


class BaseNoiseModel:
    """Abstract base class for all noise models."""

    def get_params(self) -> NoiseParams:
        raise NotImplementedError

    def apply_to_circuit(self, circuit: stim.Circuit) -> stim.Circuit:
        """Return a noisy version of a noiseless circuit."""
        raise NotImplementedError

    def step(self):
        """Advance the noise model by one time step (for drift models)."""
        pass  # Stationary models do nothing

    def reset(self):
        """Reset to initial noise parameters."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# Stationary noise models
# ────────────────────────────────────────────────────────────────────────────

class DepolarizingNoise(BaseNoiseModel):
    """
    Standard depolarizing noise. Each qubit independently suffers
    X, Y, or Z error with equal probability p/3.
    This is the textbook model and the MWPM baseline assumption.
    """

    def __init__(self, p: float = 0.005, p_meas: float = None):
        self.p = p
        self.p_meas = p_meas if p_meas is not None else p

    def get_params(self) -> NoiseParams:
        return NoiseParams(p=self.p, p_meas=self.p_meas, bias_eta=1.0)

    def step(self): pass
    def reset(self): pass


class BiasedNoise(BaseNoiseModel):
    """
    Biased-noise model. Errors are predominantly Z (phase-flip).
    Common in superconducting qubits. MWPM struggles here → RL excels.

    bias_eta: float = Z-to-X error rate ratio
        eta=1   → standard depolarizing
        eta=10  → 10x more Z errors than X errors  
        eta=inf → pure dephasing (Z-only errors)
    """

    def __init__(self, p: float = 0.005, bias_eta: float = 10.0, p_meas: float = None):
        self.p = p
        self.bias_eta = bias_eta
        self.p_meas = p_meas if p_meas is not None else p

    def get_params(self) -> NoiseParams:
        return NoiseParams(p=self.p, p_meas=self.p_meas, bias_eta=self.bias_eta)

    def step(self): pass
    def reset(self): pass


# ────────────────────────────────────────────────────────────────────────────
# Drifting noise — THE CORE RESEARCH NOVELTY
# ────────────────────────────────────────────────────────────────────────────

class DriftingNoise(BaseNoiseModel):
    """
    Wraps any stationary noise model and adds temporal drift.
    Simulates real quantum hardware, where error rates change over time
    due to temperature fluctuations, EM interference, material degradation.

    Drift types:
        "gaussian_walk" — random walk in error rate (most realistic)
        "sudden_jump"   — abrupt jumps (simulates recalibration events)
        "periodic"      — sinusoidal variation (oscillating environment)

    This is what makes MWPM fail and RL adapt.
    """

    def __init__(
        self,
        base_model: BaseNoiseModel,
        drift_type: Literal["gaussian_walk", "sudden_jump", "periodic"] = "gaussian_walk",
        drift_strength: float = 0.001,
        drift_period: int = 500,
        min_rate: float = 0.001,
        max_rate: float = 0.020,
        seed: int = 42,
    ):
        self.base_model = base_model
        self.drift_type = drift_type
        self.drift_strength = drift_strength
        self.drift_period = drift_period
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rng = np.random.default_rng(seed)

        # State
        self._step_count = 0
        self._initial_p = base_model.get_params().p
        self._current_p = self._initial_p

    @property
    def p(self) -> float:
        return self._current_p

    def get_params(self) -> NoiseParams:
        base = self.base_model.get_params()
        return NoiseParams(
            p=self._current_p,
            p_meas=self._current_p,  # co-drift measurement errors
            bias_eta=base.bias_eta,
        )

    def step(self):
        """Advance drift by one step. Called once per RL environment step."""
        self._step_count += 1

        if self.drift_type == "gaussian_walk":
            delta = self.rng.normal(0, self.drift_strength)
            self._current_p = float(np.clip(self._current_p + delta,
                                             self.min_rate, self.max_rate))

        elif self.drift_type == "sudden_jump":
            if self._step_count % self.drift_period == 0:
                self._current_p = float(self.rng.uniform(self.min_rate, self.max_rate))

        elif self.drift_type == "periodic":
            phase = 2 * np.pi * self._step_count / self.drift_period
            amplitude = (self.max_rate - self.min_rate) / 2
            center = (self.max_rate + self.min_rate) / 2
            self._current_p = float(center + amplitude * np.sin(phase))

    def reset(self):
        self._step_count = 0
        self._current_p = self._initial_p

    def get_drift_history_info(self) -> dict:
        """Expose current noise state to the RL agent (partial observability option)."""
        return {
            "current_p": self._current_p,
            "initial_p": self._initial_p,
            "step": self._step_count,
        }


# ────────────────────────────────────────────────────────────────────────────
# Noise model factory
# ────────────────────────────────────────────────────────────────────────────

def build_noise_model(cfg: dict) -> BaseNoiseModel:
    """
    Build a noise model from a config dict.

    Example cfg (from default.yaml):
        noise:
          model: "depolarizing"
          base_rate: 0.005
          drift:
            enabled: true
            type: "gaussian_walk"
            drift_strength: 0.001
    """
    model_type = cfg.get("model", "depolarizing")
    p = cfg.get("base_rate", 0.005)

    if model_type == "depolarizing":
        base = DepolarizingNoise(p=p)
    elif model_type == "biased":
        base = BiasedNoise(p=p, bias_eta=cfg.get("bias_eta", 10.0))
    else:
        raise ValueError(f"Unknown noise model: {model_type}")

    drift_cfg = cfg.get("drift", {})
    if drift_cfg.get("enabled", False):
        return DriftingNoise(
            base_model=base,
            drift_type=drift_cfg.get("type", "gaussian_walk"),
            drift_strength=drift_cfg.get("drift_strength", 0.001),
            drift_period=drift_cfg.get("drift_period", 500),
            min_rate=drift_cfg.get("min_rate", 0.001),
            max_rate=drift_cfg.get("max_rate", 0.020),
        )

    return base
