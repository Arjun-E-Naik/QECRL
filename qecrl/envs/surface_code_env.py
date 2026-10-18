"""
QECRL — Surface Code Gymnasium Environment
===========================================
The core RL training environment. Follows the Gymnasium (OpenAI Gym) API.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW THE RL ENVIRONMENT WORKS                                        ║
║                                                                      ║
║  State  = syndrome vector (binary)                                   ║
║           + optional: syndrome history (last H rounds)              ║
║           + optional: estimated current error rate                   ║
║                                                                      ║
║  Action = 0 → MWPM correction (delegate to classical)               ║
║           1 → Neural correction (use learned decoder)               ║
║           2..N → Code switch commands (surface → repetition, etc.)  ║
║                                                                      ║
║  Reward = +1   if logical qubit survives a round                     ║
║           -10  if logical qubit is flipped (major penalty)           ║
║           -0.1 per step (time penalty, encourages efficiency)        ║
╚══════════════════════════════════════════════════════════════════════╝

The KEY novelty: the noise model drifts between episodes and WITHIN
episodes. The RL agent must infer drift from syndrome statistics alone
and adapt its decoding strategy accordingly.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stim

from ..noise.noise_models import BaseNoiseModel, DepolarizingNoise, build_noise_model
from ..utils.stim_utils import (
    build_surface_code_circuit,
    build_repetition_code_circuit,
    sample_syndromes,
    syndrome_to_observation,
)
from ..decoders.mwpm_decoder import MWPMDecoder


# ────────────────────────────────────────────────────────────────────────────
# Action space definition
# ────────────────────────────────────────────────────────────────────────────

class QECAction:
    """
    Enum-style constants for the discrete action space.
    
    Phase 1 (simple):   Actions 0–1 (MWPM vs Neural correction)
    Phase 2 (research): Add Actions 2–4 (code switching)
    """
    APPLY_MWPM_CORRECTION   = 0   # Delegate to classical MWPM
    APPLY_NEURAL_CORRECTION  = 1   # Use the RL agent's own neural policy
    # Future: code switching actions
    SWITCH_TO_SURFACE_CODE   = 2
    SWITCH_TO_REPETITION_CODE = 3
    SWITCH_TO_COLOR_CODE     = 4

    N_ACTIONS_PHASE1 = 2   # Start here
    N_ACTIONS_PHASE2 = 5   # Unlock for code switching research


# ────────────────────────────────────────────────────────────────────────────
# Main environment
# ────────────────────────────────────────────────────────────────────────────

class SurfaceCodeEnv(gym.Env):
    """
    Gymnasium environment for RL-based quantum error correction
    on the rotated surface code with drifting noise.

    Inherits from gym.Env — fully compatible with Stable-Baselines3.

    Observation Space:
        Box(low=0, high=1, shape=(obs_dim,), dtype=float32)
        where obs_dim = num_detectors * (1 + syndrome_history_length)

    Action Space:
        Discrete(n_actions)
        Currently: 0 = MWPM, 1 = Neural

    Reward:
        Shaped reward encouraging logical qubit survival.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        distance: int = 3,
        rounds: int = 5,
        noise_model: BaseNoiseModel = None,
        syndrome_history_length: int = 3,
        n_actions: int = QECAction.N_ACTIONS_PHASE1,
        max_episode_steps: int = 200,
        reward_logical_survival: float = 1.0,
        reward_logical_failure: float = -10.0,
        reward_per_step: float = -0.05,
        render_mode: str = None,
    ):
        super().__init__()

        # Code parameters
        self.distance = distance
        self.rounds = rounds
        self.max_episode_steps = max_episode_steps
        self.syndrome_history_length = syndrome_history_length
        self.n_actions = n_actions

        # Rewards (tunable hyperparameters)
        self.R_SURVIVE = reward_logical_survival
        self.R_FAIL = reward_logical_failure
        self.R_STEP = reward_per_step

        # Noise model
        if noise_model is None:
            noise_model = DepolarizingNoise(p=0.005)
        self.noise_model = noise_model

        # Build initial circuit to get observation dimensions
        params = self.noise_model.get_params()
        self._circuit = build_surface_code_circuit(
            distance=distance, rounds=rounds,
            p=params.p, p_meas=params.p_meas
        )
        self._circuit_info = {
            "num_detectors": self._circuit.num_detectors,
            "num_observables": self._circuit.num_observables,
        }

        # Build MWPM decoder (action 0 delegate)
        self._mwpm = MWPMDecoder(self._circuit)

        # Gymnasium spaces
        obs_dim = self._circuit.num_detectors * (1 + syndrome_history_length)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        # Internal state
        self._syndrome_history = np.zeros(
            (syndrome_history_length, self._circuit.num_detectors), dtype=np.float32
        )
        self._step_count = 0
        self._episode_count = 0
        self._logical_failures = 0
        self._current_syndrome = None
        self._current_logical_flip = None

        self.render_mode = render_mode

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        """
        Reset environment for a new episode.
        
        Key: noise model is NOT reset — drift persists across episodes
        (simulating real hardware that drifts continuously).
        """
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_count += 1
        self._syndrome_history[:] = 0

        # Sample first syndrome from current (possibly drifted) noise
        obs = self._sample_and_update_state()
        info = self._get_info()

        return obs, info

    def step(self, action: int):
        """
        Execute one decoding step.

        Args:
            action: int in {0, ..., n_actions-1}
                    0 = MWPM correction
                    1 = neural correction (RL agent's own decision)

        Returns:
            observation: np.ndarray
            reward:      float
            terminated:  bool  (logical failure = episode ends)
            truncated:   bool  (max steps reached)
            info:        dict
        """
        self._step_count += 1

        # ── Step 1: Apply correction based on action ──────────────────────
        logical_error_after_correction = self._apply_correction(action)

        # ── Step 2: Advance noise drift ───────────────────────────────────
        self.noise_model.step()

        # ── Step 3: Rebuild circuit with new noise level ──────────────────
        self._rebuild_circuit()

        # ── Step 4: Sample new syndrome ───────────────────────────────────
        obs = self._sample_and_update_state()

        # ── Step 5: Compute reward ────────────────────────────────────────
        if logical_error_after_correction:
            reward = self.R_FAIL
            self._logical_failures += 1
            terminated = True   # Episode ends on logical failure
        else:
            reward = self.R_SURVIVE + self.R_STEP
            terminated = False

        truncated = self._step_count >= self.max_episode_steps
        info = self._get_info()
        info["logical_error"] = logical_error_after_correction
        info["action"] = action

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Internal helpers ───────────────────────────────────────────────────

    def _apply_correction(self, action: int) -> bool:
        """
        Apply correction strategy and return whether a logical error occurred.
        
        Returns:
            True if a logical error occurred (decoder failed)
            False if logical qubit survived (decoder succeeded)
        """
        syndrome = self._current_syndrome
        logical_flip = self._current_logical_flip

        if action == QECAction.APPLY_MWPM_CORRECTION:
            # Ask MWPM for the correction
            correction = self._mwpm.decode(syndrome)
            logical_error = bool(np.logical_xor(logical_flip[0], correction[0]))

        elif action == QECAction.APPLY_NEURAL_CORRECTION:
            # In Phase 1: RL agent IS the decoder via its action.
            # In Phase 2: Add a neural correction head here.
            # For now, treat as "trust the syndrome" (raw XOR approach).
            # The RL agent's VALUE comes from choosing WHEN to use MWPM vs when not.
            logical_error = bool(logical_flip[0])  # No active correction

        else:
            # Fallback for unknown actions
            logical_error = bool(logical_flip[0])

        return logical_error

    def _rebuild_circuit(self):
        """Rebuild Stim circuit with current (possibly drifted) noise params."""
        params = self.noise_model.get_params()
        self._circuit = build_surface_code_circuit(
            distance=self.distance,
            rounds=self.rounds,
            p=params.p,
            p_meas=params.p_meas,
        )
        # Note: We do NOT update MWPM here intentionally.
        # MWPM uses the OLD circuit's noise model → it drifts out of sync.
        # This is what the RL agent must overcome.

    def _sample_and_update_state(self) -> np.ndarray:
        """Sample one syndrome from the circuit and update internal state."""
        syndromes, logical_flips = sample_syndromes(self._circuit, shots=1)
        self._current_syndrome = syndromes[0]
        self._current_logical_flip = logical_flips[0]

        # Build observation: [syndrome_history | current_syndrome]
        obs = syndrome_to_observation(
            syndromes=self._current_syndrome,
            syndrome_history=self._syndrome_history,
        )

        # Update syndrome history (FIFO queue)
        self._syndrome_history = np.roll(self._syndrome_history, shift=-1, axis=0)
        self._syndrome_history[-1] = self._current_syndrome.astype(np.float32)

        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        """Return diagnostic info dict."""
        noise_params = self.noise_model.get_params()
        return {
            "step": self._step_count,
            "episode_count": self._episode_count,
            "current_error_rate": noise_params.p,
            "logical_failures": self._logical_failures,
            "distance": self.distance,
            "rounds": self.rounds,
        }

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self):
        """Simple text rendering of current state."""
        if self.render_mode == "ansi":
            params = self.noise_model.get_params()
            syndrome_weight = int(self._current_syndrome.sum()) if self._current_syndrome is not None else 0
            return (
                f"[Step {self._step_count:4d}] "
                f"Noise p={params.p:.4f} | "
                f"Syndrome weight={syndrome_weight:3d} | "
                f"Failures={self._logical_failures}"
            )

    def get_mwpm_reference_performance(self, n_shots: int = 1000) -> float:
        """
        Compute MWPM logical error rate on CURRENT circuit.
        Use this during training to track relative RL performance.
        """
        return self._mwpm.evaluate(circuit=self._circuit, n_shots=n_shots)
