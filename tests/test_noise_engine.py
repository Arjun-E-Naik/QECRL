"""
Unit Tests for Non-Stationary Noise Engine and Parameterized Surface Code Circuit Builder
"""

import unittest
import numpy as np
import stim
import pymatching

from qecrl.utils.stim_circuits import get_surface_code_layout, build_per_qubit_surface_code
from qecrl.noise.noise_engine import (
    ThermalDriftGenerator,
    TLSTelegraphGenerator,
    SpatialBurstGenerator,
    NonStationaryNoiseEngine,
)


def test_surface_code_layout():
    for d, expected_n in [(3, 17), (5, 49), (7, 97)]:
        active_qubits, coords, data_qubits, ancilla_qubits = get_surface_code_layout(d)
        assert len(active_qubits) == expected_n
        assert len(coords) == expected_n
        assert len(data_qubits) + len(ancilla_qubits) == expected_n
        assert len(data_qubits) == d * d
        assert len(ancilla_qubits) == d * d - 1


def test_per_qubit_circuit_builder():
    d = 3
    active_qubits, coords, _, _ = get_surface_code_layout(d)
    p_vec = np.full(len(active_qubits), 0.005)
    p_vec[0] = 0.02  # Make first qubit defective

    circuit = build_per_qubit_surface_code(distance=d, rounds=2, p_vec=p_vec)
    assert isinstance(circuit, stim.Circuit)
    dem = circuit.detector_error_model()
    assert dem.num_detectors > 0
    assert dem.num_errors > 0

    # Ensure PyMatching can parse it
    matcher = pymatching.Matching.from_detector_error_model(dem)
    assert matcher.num_detectors == dem.num_detectors


def test_thermal_drift_generator():
    gen = ThermalDriftGenerator(num_qubits=17, p0=0.005, p_min=0.001, p_max=0.02)
    for _ in range(50):
        p = gen.step()
        assert p.shape == (17,)
        assert np.all(p >= 0.001)
        assert np.all(p <= 0.02)


def test_tls_telegraph_generator():
    gen = TLSTelegraphGenerator(num_qubits=17, p_low=0.001, p_high=0.015, gamma_on=0.05, gamma_off=0.1)
    saw_high = False
    saw_low = False
    for _ in range(100):
        p = gen.step()
        assert p.shape == (17,)
        if np.any(p == 0.015):
            saw_high = True
        if np.any(p == 0.001):
            saw_low = True
    assert saw_high and saw_low


def test_spatial_burst_generator():
    active_qubits, coords, _, _ = get_surface_code_layout(3)
    gen = SpatialBurstGenerator(active_qubits, coords, burst_prob=0.5, surge_factor=10.0)
    saw_burst = False
    for _ in range(20):
        mult = gen.step()
        if np.any(mult > 1.0):
            saw_burst = True
            break
    assert saw_burst


def test_unified_noise_engine():
    engine = NonStationaryNoiseEngine(distance=3, regime="all", seed=42)
    for _ in range(30):
        p = engine.step()
        assert p.shape == (17,)
        assert np.all(p >= 0.0001)
        assert np.all(p <= 0.055)
    assert len(engine.history) == 30


class TestNoiseEngineUnit(unittest.TestCase):
    def test_surface_code_layout(self):
        test_surface_code_layout()

    def test_per_qubit_circuit_builder(self):
        test_per_qubit_circuit_builder()

    def test_thermal_drift_generator(self):
        test_thermal_drift_generator()

    def test_tls_telegraph_generator(self):
        test_tls_telegraph_generator()

    def test_spatial_burst_generator(self):
        test_spatial_burst_generator()

    def test_unified_noise_engine(self):
        test_unified_noise_engine()


if __name__ == "__main__":
    unittest.main()
