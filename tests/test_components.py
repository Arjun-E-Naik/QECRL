"""
Unit Tests for Decoder Components, Feature Extraction, and Bandit Agent
"""

import unittest
import numpy as np
import stim
import pymatching

from qecrl.utils.stim_circuits import get_surface_code_layout, build_per_qubit_surface_code
from qecrl.decoders.graph_updater import DynamicDEMUpdater
from qecrl.features.syndrome_features import SyndromeFeatureExtractor
from qecrl.agents.linucb_bandit import LinUCBBanditAgent


class TestComponentsUnit(unittest.TestCase):
    def setUp(self):
        self.distance = 3
        self.active_qubits, self.coords, _, _ = get_surface_code_layout(self.distance)
        self.num_qubits = len(self.active_qubits)
        self.nominal_p = np.full(self.num_qubits, 0.001)
        self.nominal_circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.distance, p_vec=self.nominal_p
        )
        self.dem = self.nominal_circuit.detector_error_model()

    def test_dynamic_dem_updater(self):
        updater = DynamicDEMUpdater(
            distance=self.distance,
            nominal_circuit=self.nominal_circuit,
            p0=0.001,
            w_min=2.85,
        )
        self.assertGreater(len(updater.raw_edges), 0)
        self.assertEqual(updater.projection_matrix.shape[1], self.num_qubits)

        # Simulate telemetry shots
        p_drift = np.copy(self.nominal_p)
        p_drift[0] = 0.035
        drift_circuit = build_per_qubit_surface_code(
            distance=self.distance, rounds=self.distance, p_vec=p_drift
        )
        syndromes, _ = drift_circuit.compile_detector_sampler().sample(
            shots=100, separate_observables=True
        )

        firing_rates = np.mean(syndromes, axis=0)
        action = np.zeros(self.num_qubits, dtype=np.float64)
        action[0] = 0.5  # moderate gain action

        matcher = updater.update_graph(action, detector_firing_rates=firing_rates)
        self.assertIsInstance(matcher, pymatching.Matching)
        self.assertEqual(matcher.num_detectors, self.dem.num_detectors)

        # Check proxy reward calculation
        reward = updater.compute_self_supervised_reward(firing_rates, action)
        self.assertIsInstance(reward, float)
        self.assertFalse(np.isnan(reward))

    def test_syndrome_feature_extractor(self):
        updater = DynamicDEMUpdater(
            distance=self.distance,
            nominal_circuit=self.nominal_circuit,
            p0=0.001,
        )
        extractor = SyndromeFeatureExtractor(
            num_detectors=self.dem.num_detectors,
            dem_edges=[(u, v) for u, v, _ in updater.raw_edges],
        )
        self.assertGreater(extractor.feature_dim, 0)

        syndromes, _ = self.nominal_circuit.compile_detector_sampler().sample(
            shots=50, separate_observables=True
        )
        features = extractor.extract(syndromes)
        self.assertEqual(features.shape, (extractor.feature_dim,))
        self.assertFalse(np.isnan(features).any())

    def test_linucb_bandit_agent(self):
        feature_dim = 32
        action_dim = self.num_qubits
        agent = LinUCBBanditAgent(
            context_dim=feature_dim,
            action_dim=action_dim,
            alpha=0.8,
        )
        context = np.random.randn(feature_dim)
        action = agent.predict(context, deterministic=False)
        self.assertEqual(action.shape, (action_dim,))

        # Verify update step
        reward = 1.25
        agent.update(context, action, reward)
        # Check action after update remains bounded
        new_action = agent.predict(context, deterministic=True)
        self.assertEqual(new_action.shape, (action_dim,))


if __name__ == "__main__":
    unittest.main()
