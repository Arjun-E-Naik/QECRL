"""
QECRL — Stim Utilities
=======================
Helpers to build Stim circuits for surface and repetition codes.

Stim encodes everything as ANNOTATED CIRCUIT INSTRUCTIONS.
We build the circuit once, then sample syndromes by sampling the circuit.

Key Stim concepts:
  stim.Circuit       — the quantum circuit (with noise)
  circuit.compile_detector_sampler()   — fast syndrome sampler
  circuit.compile_logical_sampler()    — sample logical observable outcomes
"""

from __future__ import annotations
import numpy as np
import stim
from typing import Tuple


# ────────────────────────────────────────────────────────────────────────────
# Repetition Code (simplest QEC code — good for learning & debugging)
# ────────────────────────────────────────────────────────────────────────────

def build_repetition_code_circuit(
    distance: int,
    rounds: int,
    p: float,
    p_meas: float = None,
) -> stim.Circuit:
    """
    Build a Z-basis repetition code circuit of given distance.
    Protects against X (bit-flip) errors only.

    Args:
        distance: Number of data qubits = code distance
        rounds:   Number of syndrome measurement rounds
        p:        Depolarizing error probability per gate
        p_meas:   Measurement error probability (defaults to p)

    Returns:
        stim.Circuit with noise annotations
    """
    if p_meas is None:
        p_meas = p

    # Use Stim's built-in repetition code generator
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p_meas,
        before_round_data_depolarization=p,
    )
    return circuit


def _surface_task_name(basis: str) -> str:
    """Map user-friendly basis string to Stim v1.14+ task name."""
    mapping = {
        "Z": "rotated_memory_z",
        "X": "rotated_memory_x",
        "z": "rotated_memory_z",
        "x": "rotated_memory_x",
    }
    if basis not in mapping:
        raise ValueError(f"Unknown basis '{basis}'. Use 'Z' or 'X'.")
    return mapping[basis]


# ────────────────────────────────────────────────────────────────────────────
# Surface Code (main research target)
# ────────────────────────────────────────────────────────────────────────────

def build_surface_code_circuit(
    distance: int,
    rounds: int,
    p: float,
    p_meas: float = None,
    basis: str = "Z",
) -> stim.Circuit:
    """
    Build a rotated surface code circuit.

    The rotated surface code is the INDUSTRY STANDARD for QEC.
    - Protects d=3 → 1 logical qubit using 9 data + 8 ancilla qubits
    - Threshold ~1% for depolarizing noise (best of any known 2D code)

    Args:
        distance: Code distance d (must be odd: 3, 5, 7, ...)
        rounds:   Syndrome measurement rounds per episode
        p:        Physical error probability per gate
        p_meas:   Measurement error probability
        basis:    Logical basis ("Z" or "X")

    Returns:
        stim.Circuit
    """
    if p_meas is None:
        p_meas = p

    task = f"surface_code:{_surface_task_name(basis)}"
    circuit = stim.Circuit.generated(
        task,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p_meas,
        before_round_data_depolarization=p,
    )
    return circuit


# ────────────────────────────────────────────────────────────────────────────
# Syndrome sampling — the core data loop
# ────────────────────────────────────────────────────────────────────────────

def sample_syndromes(
    circuit: stim.Circuit,
    shots: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample error syndromes and logical observable outcomes from a Stim circuit.

    A "syndrome" is the DETECTOR output: a binary vector where
      0 → no parity violation detected
      1 → parity violation (error detected nearby)

    A "logical outcome" tells us if the logical qubit was corrupted.
      0 → logical qubit survived
      1 → logical qubit was flipped (BAD — this is what we minimize)

    Args:
        circuit: Noisy stim.Circuit
        shots:   Number of independent shots to sample

    Returns:
        syndromes:      np.ndarray, shape (shots, num_detectors), dtype bool
        logical_flips:  np.ndarray, shape (shots, num_observables), dtype bool
    """
    sampler = circuit.compile_detector_sampler()
    syndromes, logical_flips = sampler.sample(shots=shots, separate_observables=True)
    return syndromes.astype(np.int8), logical_flips.astype(np.int8)


def get_circuit_info(circuit: stim.Circuit) -> dict:
    """
    Extract structural info from a Stim circuit.
    Useful for setting up RL observation/action spaces.
    """
    return {
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_instructions": len(circuit),
    }


# ────────────────────────────────────────────────────────────────────────────
# Syndrome → observation vector
# ────────────────────────────────────────────────────────────────────────────

def syndrome_to_observation(
    syndromes: np.ndarray,
    syndrome_history: np.ndarray | None = None,
    history_length: int = 3,
) -> np.ndarray:
    """
    Convert a syndrome array into an RL observation vector.

    Options:
      - Raw syndrome (flat binary vector)
      - Syndrome + history (last N rounds concatenated) — gives RL temporal info

    Args:
        syndromes:        Current syndrome, shape (num_detectors,)
        syndrome_history: Previous syndromes, shape (history_length, num_detectors)
        history_length:   How many past syndromes to include

    Returns:
        observation: np.ndarray, float32, shape (num_detectors * (1 + history_length),)
    """
    current = syndromes.astype(np.float32)
    if syndrome_history is not None:
        # Stack: [oldest, ..., newest, current]
        obs = np.concatenate([syndrome_history.flatten(), current])
    else:
        obs = current
    return obs


# ────────────────────────────────────────────────────────────────────────────
# Logical error rate estimation
# ────────────────────────────────────────────────────────────────────────────

def estimate_logical_error_rate(
    circuit: stim.Circuit,
    n_shots: int = 10_000,
    decoder_fn=None,
) -> float:
    """
    Estimate the logical error rate for a given circuit + decoder.

    Args:
        circuit:    Noisy Stim circuit
        n_shots:    Number of Monte Carlo samples
        decoder_fn: Callable(syndromes) → predicted_corrections (bool array)
                    If None, no correction is applied (raw error rate)

    Returns:
        logical_error_rate: float in [0, 0.5]
    """
    syndromes, logical_flips = sample_syndromes(circuit, shots=n_shots)

    if decoder_fn is not None:
        corrections = decoder_fn(syndromes)
        # A logical error occurs when: actual_flip XOR correction = 1
        logical_errors = np.logical_xor(logical_flips[:, 0], corrections)
    else:
        logical_errors = logical_flips[:, 0].astype(bool)

    return float(np.mean(logical_errors))


def build_detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """
    Build the Detector Error Model (DEM) from a Stim circuit.
    The DEM is what PyMatching uses for MWPM decoding.
    """
    return circuit.detector_error_model(decompose_errors=True)
