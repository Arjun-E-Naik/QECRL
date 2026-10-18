"""
QECRL — Classical Adaptive Baselines Benchmark
===============================================
Compares Static-MWPM, Spitz-Corr-MWPM, and EMA-Heuristic-MWPM against
Oracle-MWPM across varying window sizes W in {50, 100, 250, 500, 1000}.

Demonstrates the fundamental breakdown of Spitz et al. correlation inversion
under small sample sizes (W <= 250), where finite-sampling noise (sigma ~ 1/sqrt(W))
causes unstable edge weight estimates and degraded logical fidelity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pymatching

from qecrl.utils.stim_circuits import get_surface_code_layout, build_per_qubit_surface_code
from qecrl.noise.noise_engine import NonStationaryNoiseEngine
from qecrl.decoders.classical_adaptive import SpitzCorrMWPM, EMAHeuristicMWPM
from qecrl.decoders.graph_updater import DynamicDEMUpdater


def evaluate_window_size_sweep(
    distance=3,
    window_sizes=(50, 100, 250, 500, 1000),
    num_eval_windows=40,
    regime="all",
    p0=0.005,
    seed=42,
    output_dir="results",
):
    print("=" * 78)
    print("  QECRL: CLASSICAL ADAPTIVE BASELINES EVALUATION & SPITZ BREAKDOWN ANALYSIS")
    print(f"  Distance: {distance} | Window Sizes Sweep: {window_sizes}")
    print(f"  Noise Regime: '{regime}' | Windows per sweep: {num_eval_windows} | Seed: {seed}")
    print("=" * 78)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_qubits = len(get_surface_code_layout(distance)[0])
    nominal_p = np.full(num_qubits, p0, dtype=np.float64)
    nominal_circuit = build_per_qubit_surface_code(distance=distance, rounds=distance, p_vec=nominal_p)
    static_matcher = pymatching.Matching.from_detector_error_model(nominal_circuit.detector_error_model())

    sweep_results = {}

    for W in window_sizes:
        print(f"\n[Evaluating Window Size W = {W}]")
        engine = NonStationaryNoiseEngine(distance=distance, regime=regime, p0=p0, seed=seed)
        spitz = SpitzCorrMWPM(nominal_circuit)
        ema = EMAHeuristicMWPM(nominal_circuit, nominal_p=p0)

        lers_static = []
        lers_spitz = []
        lers_ema = []
        lers_rl = []
        lers_oracle = []

        updater = DynamicDEMUpdater(distance=distance, nominal_circuit=nominal_circuit, p0=p0)

        t0 = time.perf_counter()
        for w in range(num_eval_windows):
            p_vec = engine.step()
            true_circuit = build_per_qubit_surface_code(distance=distance, rounds=distance, p_vec=p_vec)
            oracle_matcher = pymatching.Matching.from_detector_error_model(true_circuit.detector_error_model())

            sampler = true_circuit.compile_detector_sampler()
            dets, obs = sampler.sample(shots=W, separate_observables=True)

            # 1. Static MWPM
            pred_static = static_matcher.decode_batch(dets)
            lers_static.append(float(np.mean(np.any(pred_static != obs, axis=1))))

            # 2. Spitz Correlation Inversion
            pred_spitz = spitz.decode_window(dets)
            lers_spitz.append(float(np.mean(np.any(pred_spitz != obs, axis=1))))

            # 3. EMA Heuristic
            pred_ema = ema.decode_window(dets)
            lers_ema.append(float(np.mean(np.any(pred_ema != obs, axis=1))))

            # 4. RL-MWPM (Ours)
            r_obs = np.mean(dets, axis=0)
            m_rl = updater.update_graph(np.full(num_qubits, 0.5), detector_firing_rates=r_obs)
            pred_rl = m_rl.decode_batch(dets)
            lers_rl.append(float(np.mean(np.any(pred_rl != obs, axis=1))))

            # 5. Oracle MWPM
            pred_oracle = oracle_matcher.decode_batch(dets)
            lers_oracle.append(float(np.mean(np.any(pred_oracle != obs, axis=1))))

        t1 = time.perf_counter()
        elapsed = t1 - t0

        res_w = {
            "mean_static": float(np.mean(lers_static)),
            "mean_spitz": float(np.mean(lers_spitz)),
            "mean_ema": float(np.mean(lers_ema)),
            "mean_rl": float(np.mean(lers_rl)),
            "mean_oracle": float(np.mean(lers_oracle)),
            "std_spitz": float(np.std(lers_spitz)),
            "elapsed_seconds": elapsed,
        }
        sweep_results[str(W)] = res_w

        print(
            f"  W={W:4d} | Static={res_w['mean_static']:.4f} | "
            f"Spitz={res_w['mean_spitz']:.4f} (std={res_w['std_spitz']:.4f}) | "
            f"EMA={res_w['mean_ema']:.4f} | RL-MWPM={res_w['mean_rl']:.4f} | "
            f"Oracle={res_w['mean_oracle']:.4f} | Time: {elapsed:.2f}s"
        )

    # ────────────────────────────────────────────────────────────────────────────
    # Render Breakdown Plot
    # ────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ws = np.array(window_sizes)
    spitz_means = [sweep_results[str(W)]["mean_spitz"] for W in ws]
    spitz_stds = [sweep_results[str(W)]["std_spitz"] for W in ws]
    ema_means = [sweep_results[str(W)]["mean_ema"] for W in ws]
    static_means = [sweep_results[str(W)]["mean_static"] for W in ws]
    rl_means = [sweep_results[str(W)]["mean_rl"] for W in ws]
    oracle_means = [sweep_results[str(W)]["mean_oracle"] for W in ws]

    ax.errorbar(ws, spitz_means, yerr=spitz_stds, fmt="-o", color="#ff7f0e", lw=2, capsize=4, label="Spitz-Corr-MWPM (Analytical Inversion)")
    ax.plot(ws, ema_means, "-s", color="#9467bd", lw=2, label="EMA-Heuristic-MWPM")
    ax.plot(ws, static_means, "--", color="#d62728", lw=1.8, label="Static-MWPM (Unadapted)")
    ax.plot(ws, rl_means, "-*", color="#1f77b4", lw=2.4, markersize=9, label="RL-MWPM (Ours, Online Adaptive)")
    ax.plot(ws, oracle_means, ":", color="#2ca02c", lw=2.2, label="Oracle-MWPM (Optimal Instantaneous)")

    ax.axvspan(0, 250, alpha=0.12, color="red", label="Low-Latency Regime (W <= 250 shots)")
    ax.set_xscale("log")
    ax.set_xlabel("Sliding Telemetry Window Size W (shots)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Logical Error Rate (LER)", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plot_file = out_path / "spitz_breakdown_analysis.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\n[Saved] Spitz breakdown plot saved to: {plot_file}")

    json_file = out_path / "spitz_breakdown_metrics.json"
    with open(json_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[Saved] Breakdown metrics JSON saved to: {json_file}")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classical Adaptive Baselines vs Window Size W")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--windows", type=int, default=30)
    parser.add_argument("--p0", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()
    evaluate_window_size_sweep(
        distance=args.distance,
        window_sizes=[50, 100, 250, 500, 1000],
        num_eval_windows=args.windows,
        p0=args.p0,
        seed=args.seed,
        output_dir=args.outdir,
    )
