"""
QECRL — MWPM Baseline Decoder
==============================
Minimum Weight Perfect Matching decoder using PyMatching.

This is the GOLD STANDARD classical decoder we benchmark against.
It assumes the noise model is KNOWN AND FIXED — which is why it fails
under drifting noise (it uses the wrong noise model for decoding).

Usage:
    decoder = MWPMDecoder(circuit)
    corrections = decoder.decode_batch(syndromes)  # shape (N,)
    logical_error_rate = decoder.evaluate(circuit, n_shots=10_000)
"""

from __future__ import annotations
import numpy as np
import stim
import pymatching
from ..utils.stim_utils import sample_syndromes, build_detector_error_model


class MWPMDecoder:
    """
    Minimum Weight Perfect Matching (MWPM) decoder.

    Theory:
        Errors create pairs of syndrome violations (detectors).
        MWPM finds the minimum-weight set of Pauli corrections
        that would cause exactly those violations.
        
    Key weakness:
        Weights are computed from a FIXED noise model.
        If actual noise drifts away from the assumed model,
        the matching weights become incorrect → decoder fails.

    This is exactly the gap our RL decoder fills.
    """

    def __init__(self, circuit: stim.Circuit):
        """
        Initialize MWPM decoder for a given circuit.

        Args:
            circuit: The Stim circuit this decoder is paired with.
                     The noise model BAKED INTO the circuit is used
                     to set MWPM edge weights.
        """
        self.circuit = circuit
        self.dem = build_detector_error_model(circuit)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode a single syndrome.

        Args:
            syndrome: Binary array of shape (num_detectors,)

        Returns:
            correction: Binary array of shape (num_observables,)
                        1 = apply logical correction, 0 = no correction
        """
        return self.matcher.decode(syndrome)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode a batch of syndromes efficiently.

        Args:
            syndromes: Binary array of shape (N, num_detectors)

        Returns:
            corrections: Binary array of shape (N, num_observables)
        """
        return self.matcher.decode_batch(syndromes)

    def evaluate(self, circuit: stim.Circuit = None, n_shots: int = 10_000) -> float:
        """
        Estimate logical error rate for this decoder on the given circuit.

        Args:
            circuit:  Circuit to evaluate on (uses self.circuit if None)
            n_shots:  Monte Carlo sample size

        Returns:
            Logical error rate (float in [0, 0.5])
        """
        eval_circuit = circuit if circuit is not None else self.circuit
        syndromes, logical_flips = sample_syndromes(eval_circuit, shots=n_shots)
        corrections = self.decode_batch(syndromes)

        # Logical error = flip XOR correction ≠ 0
        logical_errors = np.logical_xor(
            logical_flips[:, 0].astype(bool),
            corrections[:, 0].astype(bool)
        )
        return float(np.mean(logical_errors))

    def update_noise_model(self, new_circuit: stim.Circuit):
        """
        Re-initialize MWPM with a new circuit's noise model.
        
        NOTE: This is NOT adaptive — we only call this in experiments to
        show what an "oracle MWPM" (that magically knows the true noise) 
        achieves. The RL decoder must match this without being told.
        """
        self.circuit = new_circuit
        self.dem = build_detector_error_model(new_circuit)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)
