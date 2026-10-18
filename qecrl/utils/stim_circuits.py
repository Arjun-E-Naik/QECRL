"""
QECRL — Stim Circuit Generation with Per-Qubit Error Rates
===========================================================
Generates rotated surface code circuits with heterogeneous, per-qubit
physical error rates across single-qubit gates, two-qubit gates, resets,
and measurements.

Compatible with Stim v1.14+ and PyMatching v2.4+.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import stim


def get_surface_code_layout(distance: int, basis: str = "Z") -> Tuple[List[int], Dict[int, Tuple[float, float]], List[int], List[int]]:
    """
    Extract active physical qubits, their 2D grid coordinates,
    and categorize them into data qubits and ancilla (measure) qubits.

    Args:
        distance: Code distance (odd integer: 3, 5, 7, ...)
        basis: Logical basis ('Z' or 'X')

    Returns:
        active_qubits: Sorted list of active qubit indices
        coords: Dict mapping qubit index -> (x, y) coordinates
        data_qubits: List of data qubit indices
        ancilla_qubits: List of ancilla qubit indices
    """
    task = f"surface_code:rotated_memory_{basis.lower()}"
    template = stim.Circuit.generated(task, distance=distance, rounds=1)
    coords = template.get_final_qubit_coordinates()
    active_qubits = sorted(list(coords.keys()))

    # In rotated surface codes on Stim's coordinate grid:
    # Data qubits reside on odd (x, y) coordinates: x % 2 == 1 and y % 2 == 1
    # Ancilla (measure) qubits have at least one even coordinate
    data_qubits = [q for q in active_qubits if int(coords[q][0]) % 2 == 1 and int(coords[q][1]) % 2 == 1]
    ancilla_qubits = [q for q in active_qubits if q not in data_qubits]

    return active_qubits, coords, data_qubits, ancilla_qubits


def build_per_qubit_surface_code(
    distance: int,
    rounds: int,
    p_vec: np.ndarray,
    basis: str = "Z",
    flatten: bool = False,
) -> stim.Circuit:
    """
    Build a rotated surface code Stim circuit where each physical qubit q
    is subjected to its own dynamic error probability p_vec[i].

    Noise is injected physically:
      - After Reset (R): X_ERROR(p_q) on qubit q
      - After Single-qubit Clifford (H): DEPOLARIZE1(p_q) on qubit q
      - After Two-qubit Clifford (CX): DEPOLARIZE2(p_12) on pair (q1, q2) where p_12 = max(p_q1, p_q2)
      - Before Measurement (M / MR): X_ERROR(p_q) on qubit q

    Args:
        distance: Surface code distance (3, 5, 7)
        rounds: Number of syndrome extraction rounds
        p_vec: Array of shape (N_qubits,) where N_qubits = 2*d^2 - 1
        basis: 'Z' or 'X'
        flatten: If True, flattens Stim's generated REPEAT blocks so noise is
                 injected across every individual measurement round. Defaults to
                 False to maintain exact reproducibility with existing benchmark DEMs.

    Returns:
        stim.Circuit annotated with detectors and per-qubit noise channels.
    """
    task = f"surface_code:rotated_memory_{basis.lower()}"
    base_circuit = stim.Circuit.generated(
        task,
        rounds=rounds,
        distance=distance,
    )
    if flatten:
        base_circuit = base_circuit.flattened()
    coords = base_circuit.get_final_qubit_coordinates()
    active_qubits = sorted(list(coords.keys()))
    num_expected_qubits = len(active_qubits)

    if len(p_vec) != num_expected_qubits:
        raise ValueError(
            f"Expected p_vec of length {num_expected_qubits} for distance {distance}, got {len(p_vec)}"
        )

    q2idx = {q: i for i, q in enumerate(active_qubits)}
    noisy_circuit = stim.Circuit()

    for inst in base_circuit:
        name = inst.name

        # Structural and annotation instructions: preserve directly
        if name in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"):
            noisy_circuit.append(inst)

        elif name == "R":
            noisy_circuit.append(inst)
            for t in inst.targets_copy():
                q = t.value
                p = float(p_vec[q2idx[q]])
                if p > 0:
                    noisy_circuit.append("X_ERROR", [q], p)

        elif name == "H":
            noisy_circuit.append(inst)
            for t in inst.targets_copy():
                q = t.value
                p = float(p_vec[q2idx[q]])
                if p > 0:
                    noisy_circuit.append("DEPOLARIZE1", [q], p)

        elif name in ("CX", "CZ"):
            noisy_circuit.append(inst)
            targets = inst.targets_copy()
            for k in range(0, len(targets), 2):
                q1 = targets[k].value
                q2 = targets[k + 1].value
                p1 = float(p_vec[q2idx[q1]])
                p2 = float(p_vec[q2idx[q2]])
                p12 = max(p1, p2)
                if p12 > 0:
                    noisy_circuit.append("DEPOLARIZE2", [q1, q2], p12)

        elif name in ("M", "MR"):
            targets = inst.targets_copy()
            for t in targets:
                q = t.value
                p = float(p_vec[q2idx[q]])
                if p > 0:
                    noisy_circuit.append("X_ERROR", [q], p)
            noisy_circuit.append(inst)

        else:
            noisy_circuit.append(inst)

    return noisy_circuit
