"""
QECRL — Training Script
========================
Main training loop using Stable-Baselines3 PPO agent.

Run:
    python scripts/train.py
    python scripts/train.py --config config/biased_noise.yaml
    python scripts/train.py --distance 5 --total_timesteps 2000000

Monitor:
    tensorboard --logdir results/logs
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from qecrl.envs.surface_code_env import SurfaceCodeEnv
from qecrl.noise.noise_models import build_noise_model


# ────────────────────────────────────────────────────────────────────────────
# Custom callback: log quantum-specific metrics
# ────────────────────────────────────────────────────────────────────────────

class QECMetricsCallback(BaseCallback):
    """
    Custom SB3 callback that logs quantum-specific metrics to TensorBoard:
    - Current noise level (drifting p value)
    - Logical error rate vs MWPM baseline
    - Episode logical failure count
    """

    def __init__(self, eval_env: SurfaceCodeEnv, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Log current noise level
            params = self.eval_env.noise_model.get_params()
            self.logger.record("quantum/current_error_rate", params.p)

            # Log MWPM reference performance for comparison
            mwpm_ler = self.eval_env.get_mwpm_reference_performance(n_shots=500)
            self.logger.record("quantum/mwpm_logical_error_rate", mwpm_ler)

            if self.verbose:
                print(f"  [QEC] Step {self.n_calls}: p={params.p:.4f}, "
                      f"MWPM_LER={mwpm_ler:.4f}")
        return True


# ────────────────────────────────────────────────────────────────────────────
# Main training function
# ────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, args: argparse.Namespace):
    """
    Full training pipeline:
    1. Build drifting noise model
    2. Create vectorized environments
    3. Initialize PPO agent
    4. Train with callbacks
    5. Save model + results
    """

    # ── Override config with CLI args ─────────────────────────────────────
    if args.distance:
        cfg["code"]["distance"] = args.distance
    if args.total_timesteps:
        cfg["training"]["total_timesteps"] = args.total_timesteps

    # ── Paths ─────────────────────────────────────────────────────────────
    log_dir = Path(cfg["training"]["log_dir"])
    model_dir = Path(cfg["training"]["model_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Build noise model ─────────────────────────────────────────────────
    noise_model = build_noise_model(cfg["noise"])
    print(f"[QECRL] Noise model: {type(noise_model).__name__}")
    print(f"[QECRL] Initial error rate: p={noise_model.get_params().p:.4f}")

    # ── Create training environment ───────────────────────────────────────
    def make_env():
        noise = build_noise_model(cfg["noise"])  # each env gets its own noise
        env = SurfaceCodeEnv(
            distance=cfg["code"]["distance"],
            rounds=cfg["code"]["rounds"],
            noise_model=noise,
            max_episode_steps=200,
        )
        return Monitor(env)

    n_envs = 4  # Parallel envs for faster data collection
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=cfg["training"]["seed"])

    # ── Create evaluation environment ─────────────────────────────────────
    eval_noise = build_noise_model(cfg["noise"])
    eval_env = SurfaceCodeEnv(
        distance=cfg["code"]["distance"],
        rounds=cfg["code"]["rounds"],
        noise_model=eval_noise,
    )
    eval_env = Monitor(eval_env)

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=cfg["training"]["eval_freq"] // n_envs,
        n_eval_episodes=cfg["training"]["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["training"]["save_freq"] // n_envs,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="qecrl_ppo",
        verbose=1,
    )

    qec_callback = QECMetricsCallback(
        eval_env=eval_env,
        eval_freq=cfg["training"]["eval_freq"] // n_envs,
        verbose=1,
    )

    # ── Initialize PPO agent ──────────────────────────────────────────────
    agent_cfg = cfg["agent"]
    model = PPO(
        policy=agent_cfg["policy"],
        env=vec_env,
        learning_rate=agent_cfg["learning_rate"],
        n_steps=agent_cfg["n_steps"],
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        policy_kwargs={"net_arch": agent_cfg["net_arch"]},
        tensorboard_log=str(log_dir),
        seed=cfg["training"]["seed"],
        verbose=1,
    )

    print("\n" + "="*60)
    print("  QECRL Training Starting")
    print("="*60)
    print(f"  Code:       Surface code d={cfg['code']['distance']}")
    print(f"  Rounds:     {cfg['code']['rounds']} per episode")
    print(f"  Algorithm:  PPO")
    print(f"  Timesteps:  {cfg['training']['total_timesteps']:,}")
    print(f"  Log dir:    {log_dir}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print("="*60 + "\n")

    # Check if rich & tqdm are installed for SB3 progress bar
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
        has_progress_bar = True
    except ImportError:
        has_progress_bar = False

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, qec_callback],
        progress_bar=has_progress_bar,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = model_dir / "qecrl_final"
    model.save(str(final_path))
    print(f"\n[QECRL] Training complete. Model saved to {final_path}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train QECRL PPO agent")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--distance", type=int, default=None,
                        help="Override code distance")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override training timesteps")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)


if __name__ == "__main__":
    main()
