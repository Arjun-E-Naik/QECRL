"""
Benchmark script: Pure Algorithmic Outer-Loop Latency vs. Python Object Reconstruction
and In-Place / Double-Buffered Graph Weight Update.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import platform
import psutil
import numpy as np
import stim
import pymatching
from scipy.optimize import nnls

from qecrl.utils.stim_circuits import build_per_qubit_surface_code, get_surface_code_layout
from qecrl.features.syndrome_features import SyndromeFeatureExtractor
from qecrl.decoders.graph_updater import DynamicDEMUpdater
from qecrl.agents.linucb_bandit import LinUCBBanditAgent

def get_cpu_info():
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
    }
    return info

def benchmark_latencies():
    print("=" * 80)
    print("  QECRL OUTER-LOOP LATENCY DISAGGREGATION & IN-PLACE MUTABILITY BENCHMARK")
    print("=" * 80)
    cpu_info = get_cpu_info()
    print(f"System: {cpu_info['system']} | Machine: {cpu_info['machine']}")
    print(f"Processor: {cpu_info['processor']}")
    print(f"Cores: {cpu_info['cpu_count_physical']} physical, {cpu_info['cpu_count_logical']} logical | Freq: {cpu_info['cpu_freq_mhz']} MHz\n")

    results = {}

    for d in (3, 5, 7):
        print(f"--- Benchmarking Distance d={d} ---")
        qubits, coords, data_q, anc_q = get_surface_code_layout(d)
        num_qubits = len(qubits)
        rounds = d

        nominal_circuit = build_per_qubit_surface_code(d, rounds, np.full(num_qubits, 0.001))
        dem = nominal_circuit.detector_error_model()
        num_detectors = dem.num_detectors
        num_edges = len(pymatching.Matching.from_detector_error_model(dem).edges())

        updater = DynamicDEMUpdater(
            distance=d,
            nominal_circuit=nominal_circuit,
            p0=0.001,
            w_min=2.85,
            alpha_threshold=0.12 if d <= 3 else 0.18,
            score_threshold=0.10 if d <= 3 else 0.15,
        )

        fe = SyndromeFeatureExtractor(num_detectors, [(u, v) for u, v, _ in updater.raw_edges])
        bandit = LinUCBBanditAgent(context_dim=fe.feature_dim, action_dim=num_qubits, alpha=1.2)

        # Generate representative telemetry data (W=250 shots)
        rng = np.random.default_rng(42)
        dets_tel = rng.binomial(1, 0.02, size=(250, num_detectors)).astype(np.uint8)
        r_inst = np.mean(dets_tel, axis=0)

        # Warm up
        ctx = fe.extract(dets_tel)
        updater._update_temporal_filter(r_inst)
        bandit.update(ctx, np.zeros(num_qubits), 5.0)

        # 1. Component 1: Syndrome Feature Extraction
        N_TRIALS = 200
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            ctx = fe.extract(dets_tel)
        t_fe = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # 2. Component 2: Temporal CUSUM-EMA Filter
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            filt_rates = updater._update_temporal_filter(r_inst)
        t_ema = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # 3. Component 3: Sparse NNLS Deconvolution
        res = np.maximum(0.0, filt_rates - updater.r0)
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            alpha_raw, _ = nnls(updater.D_data, res)
        t_nnls = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # 4. Component 4: Sherman-Morrison LinUCB Prediction & Rank-1 Update
        act = bandit.predict(ctx)
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            bandit.update(ctx, act, 5.0)
            act = bandit.predict(ctx)
        t_bandit = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # 5. Component 5: Analytical Prior + Continuous Sensitivity Projection Pi_cont * Delta_w
        p_hat = updater.p0 + alpha_raw * (updater.p_cal - updater.p0)
        w_nom_q = np.log((1.0 - updater.p0) / updater.p0)
        w_hat_q = np.log((1.0 - np.clip(p_hat, updater.p0, 0.065)) / np.clip(p_hat, updater.p0, 0.065))
        bayes_delta_q = np.maximum(0.0, w_nom_q - w_hat_q)
        rl_gain = np.clip(1.0 + 0.40 * act, 0.60, 1.45)
        qubit_shifts = bayes_delta_q * rl_gain

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            e_shifts = np.minimum(updater.projection_matrix @ qubit_shifts, np.max(qubit_shifts))
            up_w = np.clip(updater.nominal_weights - e_shifts, updater.w_min, updater.w_max)
        t_proj = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # Pure algorithmic outer-loop execution time
        t_pure_algo = t_fe + t_ema + t_nnls + t_bandit + t_proj

        # 6. Python Graph Re-instantiation overhead (instantiating new Matching object and looping)
        N_RECON = 20
        t0 = time.perf_counter()
        for _ in range(N_RECON):
            m = pymatching.Matching()
            for idx, (u, v, data) in enumerate(updater.raw_edges):
                w = float(up_w[idx])
                f = data["fault_ids"]
                if v is None:
                    m.add_boundary_edge(u, weight=w, fault_ids=f)
                else:
                    m.add_edge(u, v, weight=w, fault_ids=f)
        t_py_reconstruct = (time.perf_counter() - t0) / N_RECON * 1e6

        # 7. In-Place C++ Double-Buffered Weight Swap Simulation
        # In C++, weights array is updated in memory in background thread, then pointer is swapped
        # We time updating a pre-allocated numpy weights buffer and pointer assignment
        weights_buffer_1 = updater.nominal_weights.copy()
        weights_buffer_2 = updater.nominal_weights.copy()
        active_buffer_ref = [weights_buffer_1]

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            # SIMD-like array copy into background buffer
            np.copyto(weights_buffer_2, up_w)
            # Pointer / reference swap
            active_buffer_ref[0] = weights_buffer_2
        t_buffer_swap = (time.perf_counter() - t0) / N_TRIALS * 1e6

        # Per-round inner decode latency
        # 1 shot = d stabilizer rounds
        # Inner decode times from Table I:
        inner_shot_us = {3: 0.33, 5: 1.15, 7: 2.45}[d]
        inner_round_us = inner_shot_us / rounds

        print(f"Detectors: {num_detectors} | Graph Edges: {num_edges}")
        print(f"  [1] Feature Extraction:        {t_fe:6.2f} us")
        print(f"  [2] CUSUM-EMA Filter:          {t_ema:6.2f} us")
        print(f"  [3] Active-Set NNLS:           {t_nnls:6.2f} us")
        print(f"  [4] Sherman-Morrison LinUCB:   {t_bandit:6.2f} us")
        print(f"  [5] Pi_cont Matrix Projection: {t_proj:6.2f} us")
        print(f"  => PURE ALGORITHMIC LATENCY:   {t_pure_algo:6.2f} us ({t_pure_algo/1000:6.3f} ms)")
        print(f"  [Python Rebuild Overhead]:     {t_py_reconstruct:6.2f} us ({t_py_reconstruct/1000:6.2f} ms)")
        print(f"  [C++ Double-Buffer Swap]:      {t_buffer_swap:6.2f} us")
        print(f"  [Inner Decode Latency]:        {inner_shot_us:.2f} us/shot -> {inner_round_us:.3f} us/round (Budget: 1.000 us/round)\n")

        results[d] = {
            "num_detectors": num_detectors,
            "num_edges": num_edges,
            "t_fe_us": t_fe,
            "t_ema_us": t_ema,
            "t_nnls_us": t_nnls,
            "t_bandit_us": t_bandit,
            "t_proj_us": t_proj,
            "t_pure_algo_us": t_pure_algo,
            "t_py_reconstruct_ms": t_py_reconstruct / 1000.0,
            "t_buffer_swap_us": t_buffer_swap,
            "inner_shot_us": inner_shot_us,
            "inner_round_us": inner_round_us,
        }

    import json
    with open("results/latency_disaggregation_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Saved] Latency benchmark saved to results/latency_disaggregation_benchmark.json")

if __name__ == "__main__":
    benchmark_latencies()
