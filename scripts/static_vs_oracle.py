"""
QECRL — Static-MWPM vs. Oracle-MWPM Benchmark (Figure 1 Generator)
===================================================================
Demonstrates the fundamental breakdown of static matching decoders under
non-stationary noise (Regime B: Two-Level System Telegraph bursts, etc.).

Compares:
  1. Static-MWPM: Standard PyMatching calibrated at t=0 (never updated)
  2. Oracle-MWPM: PyMatching calibrated with exact instantaneous p_q(t)

Generates Figure 1 of the manuscript:
  - Panel 1: Physical Noise Dynamics p(t) across Qubits
  - Panel 2: Logical Error Rate for Distance d=3
  - Panel 3: Logical Error Rate for Distance d=5
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


def run_static_vs_oracle_benchmark(
    distances=(3, 5),
    num_windows=60,
    shots_per_window=2000,
    regime="all",
    p0=0.001,
    seed=42,
    output_dir="results",
):
    print("=" * 78)
    print("  QECRL BENCHMARK: STATIC-MWPM vs. ORACLE-MWPM UNDER NON-STATIONARY NOISE")
    print(f"  Distances: {distances} | Windows: {num_windows} | Shots/Window: {shots_per_window}")
    print(f"  Noise Regime: '{regime}' | Baseline p0: {p0} | Seed: {seed}")
    print("=" * 78)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "distances": list(distances),
        "num_windows": num_windows,
        "shots_per_window": shots_per_window,
        "regime": regime,
        "p0": p0,
        "metrics": {},
    }

    # Store trajectories
    trajectories = {}

    for d in distances:
        print(f"\n[Benchmarking Distance d={d}]")
        rounds = d
        engine = NonStationaryNoiseEngine(distance=d, regime=regime, p0=p0, seed=seed)
        num_qubits = engine.num_qubits

        # Initialize Static-MWPM with nominal t=0 model
        nominal_p_vec = np.full(num_qubits, p0, dtype=np.float64)
        static_circuit = build_per_qubit_surface_code(distance=d, rounds=rounds, p_vec=nominal_p_vec)
        static_dem = static_circuit.detector_error_model()
        static_matcher = pymatching.Matching.from_detector_error_model(static_dem)

        static_lers = []
        oracle_lers = []
        p_means = []
        p_maxs = []

        t0_d = time.perf_counter()

        for w in range(num_windows):
            p_vec = engine.step()
            p_means.append(float(np.mean(p_vec)))
            p_maxs.append(float(np.max(p_vec)))

            # Build true circuit and compile Oracle-MWPM
            true_circuit = build_per_qubit_surface_code(distance=d, rounds=rounds, p_vec=p_vec)
            oracle_dem = true_circuit.detector_error_model()
            oracle_matcher = pymatching.Matching.from_detector_error_model(oracle_dem)

            # Sample W shots from true non-stationary channel
            sampler = true_circuit.compile_detector_sampler()
            dets, obs = sampler.sample(shots=shots_per_window, separate_observables=True)

            # 1. Decode with Static-MWPM (stale t=0 weights)
            static_preds = static_matcher.decode_batch(dets)
            ler_static = float(np.mean(np.any(static_preds != obs, axis=1)))
            static_lers.append(ler_static)

            # 2. Decode with Oracle-MWPM (exact instantaneous weights)
            oracle_preds = oracle_matcher.decode_batch(dets)
            ler_oracle = float(np.mean(np.any(oracle_preds != obs, axis=1)))
            oracle_lers.append(ler_oracle)

            if (w + 1) % 30 == 0 or w == 0:
                print(
                    f"  Window {w+1:3d}/{num_windows} | "
                    f"p_mean={p_means[-1]:.4f}, p_max={p_maxs[-1]:.4f} | "
                    f"Static LER={ler_static:.4f} | Oracle LER={ler_oracle:.4f}"
                )

        t1_d = time.perf_counter()
        elapsed = t1_d - t0_d
        print(f"  Finished d={d} in {elapsed:.2f}s ({elapsed/num_windows*1000:.1f} ms/window)")

        static_arr = np.array(static_lers)
        oracle_arr = np.array(oracle_lers)
        mean_static = float(np.mean(static_arr))
        mean_oracle = float(np.mean(oracle_arr))
        peak_ratio = float(np.max(static_arr / np.maximum(oracle_arr, 1e-5)))

        results["metrics"][f"d={d}"] = {
            "mean_static_ler": mean_static,
            "mean_oracle_ler": mean_oracle,
            "overall_ler_gap": mean_static - mean_oracle,
            "peak_static_to_oracle_ratio": peak_ratio,
        }

        trajectories[d] = {
            "static_lers": static_lers,
            "oracle_lers": oracle_lers,
            "p_means": p_means,
            "p_maxs": p_maxs,
        }

    # Compute error suppression factor Lambda = LER(d=3) / LER(d=5)
    if 3 in distances and 5 in distances:
        ler_s3 = np.array(trajectories[3]["static_lers"])
        ler_s5 = np.array(trajectories[5]["static_lers"])
        ler_o3 = np.array(trajectories[3]["oracle_lers"])
        ler_o5 = np.array(trajectories[5]["oracle_lers"])

        lambda_static = float(np.mean(ler_s3 / np.maximum(ler_s5, 1e-5)))
        lambda_oracle = float(np.mean(ler_o3 / np.maximum(ler_o5, 1e-5)))
        results["metrics"]["lambda_suppression"] = {
            "lambda_static_mean": lambda_static,
            "lambda_oracle_mean": lambda_oracle,
            "interpretation": "Lambda > 1 means increasing distance suppresses error (below threshold)",
        }
        print("\n" + "=" * 78)
        print(f"  Error Suppression Factor Lambda (d=3 -> d=5):")
        print(f"    Static-MWPM Lambda: {lambda_static:.3f}")
        print(f"    Oracle-MWPM Lambda: {lambda_oracle:.3f}")
        print("=" * 78)

    # ────────────────────────────────────────────────────────────────────────────
    # Render Publication-Grade Figure 1
    # ────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        "Figure 1: Breakdown of Static MWPM vs. Oracle MWPM under Non-Stationary Noise\n"
        "(Rotated Planar Surface Code under Thermal Drift, TLS Telegraph, and Burst Crosstalk)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    windows_axis = np.arange(1, num_windows + 1)
    # 1. Top Panel: Physical Noise Dynamics
    p_mean_d3 = trajectories[3]["p_means"] if 3 in trajectories else trajectories[distances[0]]["p_means"]
    p_max_d3 = trajectories[3]["p_maxs"] if 3 in trajectories else trajectories[distances[0]]["p_maxs"]

    axes[0].plot(windows_axis, p_mean_d3, color="#1f77b4", lw=2.0, label="Mean Physical Error Rate ⟨p_q(t)⟩")
    axes[0].plot(windows_axis, p_max_d3, color="#ff7f0e", lw=1.8, ls="--", label="Max Defect Error Rate (TLS / Burst)")
    axes[0].axhline(p0, color="gray", ls=":", lw=1.2, label=rf"Nominal Calibration $p_0 = {p0}$")
    axes[0].set_ylabel("Physical Error Rate p", fontsize=11, fontweight="semibold")
    axes[0].set_title("(a) Non-Stationary Hardware Channel Dynamics (Regimes A, B, C)", fontsize=11)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Rolling window smoothing helper
    def rolling_smooth(arr, k=5):
        return np.convolve(arr, np.ones(k)/k, mode="same")

    # 2. Middle Panel: Logical Error Rate d=3
    if 3 in trajectories:
        s3 = np.array(trajectories[3]["static_lers"])
        o3 = np.array(trajectories[3]["oracle_lers"])
        axes[1].plot(windows_axis, s3, color="#d62728", alpha=0.35, lw=1.0)
        axes[1].plot(windows_axis, o3, color="#2ca02c", alpha=0.35, lw=1.0)
        axes[1].plot(windows_axis, rolling_smooth(s3), color="#d62728", lw=2.2, label="Static-MWPM (Stale DEM)")
        axes[1].plot(windows_axis, rolling_smooth(o3), color="#2ca02c", lw=2.2, label="Oracle-MWPM (Instantaneous DEM)")
        axes[1].set_ylabel("Logical Error Rate (d=3)", fontsize=11, fontweight="semibold")
        axes[1].set_title("(b) Logical Performance at Code Distance d=3", fontsize=11)
        axes[1].legend(loc="upper left", frameon=True, fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Bottom Panel: Logical Error Rate d=5
    target_d = 5 if 5 in trajectories else distances[-1]
    sd = np.array(trajectories[target_d]["static_lers"])
    od = np.array(trajectories[target_d]["oracle_lers"])
    axes[2].plot(windows_axis, sd, color="#d62728", alpha=0.35, lw=1.0)
    axes[2].plot(windows_axis, od, color="#2ca02c", alpha=0.35, lw=1.0)
    axes[2].plot(windows_axis, rolling_smooth(sd), color="#d62728", lw=2.2, label=f"Static-MWPM (d={target_d})")
    axes[2].plot(windows_axis, rolling_smooth(od), color="#2ca02c", lw=2.2, label=f"Oracle-MWPM (d={target_d})")
    axes[2].set_ylabel(f"Logical Error Rate (d={target_d})", fontsize=11, fontweight="semibold")
    axes[2].set_xlabel("Sliding Window Index t (W = 250-300 shots per window)", fontsize=11, fontweight="semibold")
    axes[2].set_title(f"(c) Logical Performance at Code Distance d={target_d} (Amplified Degradation)", fontsize=11)
    axes[2].legend(loc="upper left", frameon=True, fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_file = out_path / "figure1_static_vs_oracle.png"
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Figure 1 plot saved to: {fig_file}")

    # Save metrics JSON
    json_file = out_path / "figure1_metrics.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Metrics JSON saved to: {json_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Static-MWPM vs Oracle-MWPM Benchmark (Figure 1)")
    parser.add_argument("--windows", type=int, default=50, help="Number of sliding windows")
    parser.add_argument("--shots", type=int, default=2000, help="Syndrome shots per window")
    parser.add_argument("--regime", type=str, default="all", choices=["all", "thermal", "tls", "burst"])
    parser.add_argument("--p0", type=float, default=0.001, help="Baseline physical error rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")

    args = parser.parse_args()
    run_static_vs_oracle_benchmark(
        distances=(3, 5),
        num_windows=args.windows,
        shots_per_window=args.shots,
        regime=args.regime,
        p0=args.p0,
        seed=args.seed,
        output_dir=args.outdir,
    )
