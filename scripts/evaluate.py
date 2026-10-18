"""
QECRL — Evaluation & Benchmarking Script
==========================================
Compare RL agent vs MWPM baseline across noise conditions.

Run:
    python scripts/evaluate.py --model results/models/best/best_model.zip
    python scripts/evaluate.py --sweep_distance   # Threshold analysis
    python scripts/evaluate.py --sweep_drift       # Drift robustness
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO

from qecrl.envs.surface_code_env import SurfaceCodeEnv, QECAction
from qecrl.noise.noise_models import DepolarizingNoise, DriftingNoise
from qecrl.decoders.mwpm_decoder import MWPMDecoder
from qecrl.utils.stim_utils import (
    build_surface_code_circuit,
    estimate_logical_error_rate,
    sample_syndromes,
)


# ────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    model: PPO,
    env: SurfaceCodeEnv,
    n_episodes: int = 1000,
    deterministic: bool = True,
) -> dict:
    """
    Run RL agent on environment and compute:
    - Mean episode length (longer = better — survived more rounds)
    - Logical error rate (fraction of episodes ending in logical failure)
    - Mean reward
    """
    episode_lengths = []
    logical_failures = []
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_length = 0
        ep_reward = 0.0
        ep_failure = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
            if terminated and info.get("logical_error", False):
                ep_failure = True

        episode_lengths.append(ep_length)
        logical_failures.append(ep_failure)
        total_rewards.append(ep_reward)

    return {
        "mean_episode_length": float(np.mean(episode_lengths)),
        "logical_error_rate":  float(np.mean(logical_failures)),
        "mean_reward":         float(np.mean(total_rewards)),
        "std_episode_length":  float(np.std(episode_lengths)),
        "n_episodes":          n_episodes,
    }


# ────────────────────────────────────────────────────────────────────────────
# Drift robustness comparison (THE KEY EXPERIMENT)
# ────────────────────────────────────────────────────────────────────────────

def run_drift_robustness_experiment(
    model_path: str,
    distance: int = 3,
    rounds: int = 5,
    n_steps: int = 2000,
    n_shots_per_step: int = 200,
):
    """
    THE MAIN EXPERIMENT:
    Simulate drift over time and compare:
      1. MWPM with ORIGINAL (wrong) noise model — degrades as drift increases
      2. MWPM with ORACLE (correct) noise model — upper bound
      3. RL Agent — should track the oracle without being told the noise level

    This plot is the CORE FIGURE of the research paper.
    """
    print("\n[Eval] Running drift robustness experiment...")

    # Setup
    base_p = 0.005
    base_noise = DepolarizingNoise(p=base_p)
    drifting_noise = DriftingNoise(
        base_model=base_noise,
        drift_type="gaussian_walk",
        drift_strength=0.001,
        min_rate=0.001,
        max_rate=0.015,
    )

    # Load RL model
    model = PPO.load(model_path)

    # Data storage
    steps = []
    true_p_vals = []
    mwpm_static_ler = []   # MWPM with original noise model
    mwpm_oracle_ler = []   # MWPM with true (drifted) noise model (oracle)
    rl_episode_lengths = []  # RL agent survival (proxy for LER)

    # Initial circuit + MWPM (uses ORIGINAL p)
    static_circuit = build_surface_code_circuit(distance, rounds, p=base_p)
    mwpm_static = MWPMDecoder(static_circuit)

    # Build RL env
    rl_env = SurfaceCodeEnv(
        distance=distance, rounds=rounds,
        noise_model=DriftingNoise(
            base_model=DepolarizingNoise(p=base_p),
            drift_type="gaussian_walk", drift_strength=0.001,
        )
    )

    for step in range(n_steps):
        drifting_noise.step()
        current_p = drifting_noise.p

        # Build circuit with true current noise
        true_circuit = build_surface_code_circuit(distance, rounds, p=current_p)
        mwpm_oracle = MWPMDecoder(true_circuit)

        # ── MWPM Static: uses wrong (original) noise model ──
        syndromes, logical_flips = sample_syndromes(true_circuit, shots=n_shots_per_step)
        corrections_static = mwpm_static.decode_batch(syndromes)
        ler_static = float(np.mean(np.logical_xor(
            logical_flips[:, 0].astype(bool),
            corrections_static[:, 0].astype(bool)
        )))

        # ── MWPM Oracle: uses true noise model ──
        corrections_oracle = mwpm_oracle.decode_batch(syndromes)
        ler_oracle = float(np.mean(np.logical_xor(
            logical_flips[:, 0].astype(bool),
            corrections_oracle[:, 0].astype(bool)
        )))

        # ── RL Agent: evaluated in env ──
        obs, _ = rl_env.reset()
        done = False
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = rl_env.step(int(action))
            ep_len += 1
            done = terminated or truncated
        rl_episode_lengths.append(ep_len)

        steps.append(step)
        true_p_vals.append(current_p)
        mwpm_static_ler.append(ler_static)
        mwpm_oracle_ler.append(ler_oracle)

        if step % 200 == 0:
            print(f"  Step {step:4d} | p={current_p:.4f} | "
                  f"MWPM_static={ler_static:.4f} | "
                  f"MWPM_oracle={ler_oracle:.4f} | "
                  f"RL_ep_len={ep_len}")

    # ── Plot results ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "QECRL: RL Decoder vs MWPM under Drifting Noise\n"
        f"Surface Code d={distance}, {rounds} rounds",
        fontsize=14, fontweight="bold"
    )

    # Top panel: noise drift trajectory
    axes[0].plot(steps, true_p_vals, color="gray", lw=1.5, label="True error rate p(t)")
    axes[0].axhline(base_p, color="black", ls="--", lw=1, label=f"Initial p={base_p}")
    axes[0].set_ylabel("Physical Error Rate p", fontsize=12)
    axes[0].set_xlabel("Simulation Step", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_title("Noise Drift Over Time", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: decoder performance comparison
    axes[1].plot(steps, mwpm_static_ler, color="red", lw=1.5,
                 label="MWPM (static, wrong model)", alpha=0.8)
    axes[1].plot(steps, mwpm_oracle_ler, color="green", lw=1.5,
                 label="MWPM (oracle, true model)", alpha=0.8)
    # Normalize RL episode lengths to a "survival rate" proxy
    max_len = 200
    rl_survival = np.array(rl_episode_lengths) / max_len
    axes[1].plot(steps, 1 - rl_survival, color="blue", lw=2.0,
                 label="RL Agent (adaptive)", alpha=0.9)
    axes[1].set_ylabel("Logical Error Rate", fontsize=12)
    axes[1].set_xlabel("Simulation Step", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_title("Decoder Performance Under Drift", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("results") / "drift_robustness.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Eval] Plot saved to {out_path}")

    return {
        "steps": steps,
        "true_p": true_p_vals,
        "mwpm_static_ler": mwpm_static_ler,
        "mwpm_oracle_ler": mwpm_oracle_ler,
        "rl_episode_lengths": rl_episode_lengths,
    }


# ────────────────────────────────────────────────────────────────────────────
# Quick sanity-check: no model needed (MWPM only)
# ────────────────────────────────────────────────────────────────────────────

def run_mwpm_baseline_only(distance: int = 3, rounds: int = 5):
    """
    Run MWPM-only evaluation across error rates.
    Use this BEFORE training to understand the baseline.
    """
    print(f"\n[Baseline] MWPM threshold analysis (d={distance}, rounds={rounds})")
    error_rates = [0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]
    results = []

    for p in error_rates:
        circuit = build_surface_code_circuit(distance, rounds, p=p)
        mwpm = MWPMDecoder(circuit)
        ler = mwpm.evaluate(n_shots=5000)
        print(f"  p={p:.3f} → Logical Error Rate = {ler:.5f}")
        results.append({"p": p, "ler": ler})

    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate QECRL agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained SB3 model (.zip)")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run MWPM baseline only (no trained model needed)")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.baseline_only or args.model is None:
        run_mwpm_baseline_only(distance=args.distance, rounds=args.rounds)
    else:
        results = run_drift_robustness_experiment(
            model_path=args.model,
            distance=args.distance,
            rounds=args.rounds,
        )
        # Save results as JSON
        out = Path("results") / "drift_experiment.json"
        with open(out, "w") as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                       for k, v in results.items()}, f, indent=2)
        print(f"[Eval] Results saved to {out}")


if __name__ == "__main__":
    main()
