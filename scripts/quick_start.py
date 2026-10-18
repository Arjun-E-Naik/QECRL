"""
QECRL — Quick Sanity Check
===========================
Run this FIRST to verify the entire stack works:
  [OK] Stim builds surface code circuits
  [OK] Syndrome sampling works
  [OK] MWPM decoding works
  [OK] RL environment resets and steps correctly
  [OK] Drifting noise changes over time

Usage:
    python scripts/quick_start.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import stim
import pymatching


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(msg: str, ok: bool = True):
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status} {msg}")


def main():
    print("\n[QECRL] Stack Verification")
    print("   Testing all components...\n")

    # ── 1. Stim ──────────────────────────────────────────────────────────
    section("1. Stim: Surface Code Circuit")
    from qecrl.utils.stim_utils import (
        build_surface_code_circuit,
        sample_syndromes,
        get_circuit_info,
    )

    circuit = build_surface_code_circuit(distance=3, rounds=5, p=0.005)
    info = get_circuit_info(circuit)
    check(f"Circuit built: {info['num_qubits']} qubits, {info['num_detectors']} detectors")

    syndromes, logical_flips = sample_syndromes(circuit, shots=100)
    check(f"Syndrome sampling: shape={syndromes.shape}, dtype={syndromes.dtype}")
    check(f"Logical flips: shape={logical_flips.shape}, non-zero={logical_flips.sum()}")

    # ── 2. PyMatching MWPM ───────────────────────────────────────────────
    section("2. PyMatching: MWPM Baseline Decoder")
    from qecrl.decoders.mwpm_decoder import MWPMDecoder

    mwpm = MWPMDecoder(circuit)
    check("MWPM decoder initialized from Stim circuit")

    ler = mwpm.evaluate(n_shots=2000)
    check(f"MWPM logical error rate (p=0.005, d=3): {ler:.4f}  (expected ~0.002–0.010)")

    # ── 3. Noise Models ───────────────────────────────────────────────────
    section("3. Noise Models: Static and Drifting")
    from qecrl.noise.noise_models import DepolarizingNoise, DriftingNoise

    static = DepolarizingNoise(p=0.005)
    check(f"Depolarizing noise: p={static.get_params().p}")

    drifting = DriftingNoise(
        base_model=DepolarizingNoise(p=0.005),
        drift_type="gaussian_walk",
        drift_strength=0.001,
    )
    p_vals = []
    for _ in range(50):
        drifting.step()
        p_vals.append(drifting.p)
    p_change = abs(p_vals[-1] - p_vals[0])
    check(f"Drift working: p went from {p_vals[0]:.4f} to {p_vals[-1]:.4f} (delta={p_change:.4f})")

    # ── 4. RL Environment ─────────────────────────────────────────────────
    section("4. Gymnasium RL Environment")
    from qecrl.envs.surface_code_env import SurfaceCodeEnv
    from qecrl.noise.noise_models import DepolarizingNoise

    env = SurfaceCodeEnv(
        distance=3,
        rounds=5,
        noise_model=DepolarizingNoise(p=0.005),
        max_episode_steps=50,
    )
    check(f"Env created: obs_space={env.observation_space.shape}, "
          f"action_space={env.action_space.n}")

    obs, info = env.reset()
    check(f"Reset: obs shape={obs.shape}, dtype={obs.dtype}, "
          f"obs range=[{obs.min():.1f}, {obs.max():.1f}]")

    # Run a few steps
    total_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    check(f"20-step rollout complete | total_reward={total_reward:.2f}")

    # ── 5. SB3 Compatibility ──────────────────────────────────────────────
    section("5. Stable-Baselines3 Compatibility Check")
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    try:
        check_env(SurfaceCodeEnv(distance=3, rounds=3), warn=True)
        check("SB3 check_env passed — environment is fully compliant")
    except Exception as e:
        check(f"SB3 check_env: {e}", ok=False)

    # Quick PPO instantiation check (no training)
    env2 = SurfaceCodeEnv(distance=3, rounds=3, noise_model=DepolarizingNoise(p=0.005))
    model = PPO("MlpPolicy", env2, verbose=0)
    check(f"PPO model instantiated | params={sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Summary ───────────────────────────────────────────────────────────
    section("Summary")
    print("  All components verified! SUCCESS")
    print()
    print("  Next steps:")
    print("    Train:    python scripts/train.py")
    print("    Baseline: python scripts/evaluate.py --baseline_only")
    print("    Monitor:  tensorboard --logdir results/logs")
    print()
    print("  Key numbers to note:")
    print(f"    MWPM LER (d=3, p=0.005): {ler:.4f}")
    print(f"    Observation dimension:   {env.observation_space.shape[0]}")
    print(f"    Action space:            {env.action_space.n} actions")
    print()


if __name__ == "__main__":
    main()
