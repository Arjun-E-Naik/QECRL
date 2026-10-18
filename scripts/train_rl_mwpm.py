"""
QECRL — RL-MWPM Meta-Controller Training
=========================================
Trains continuous-action RL agents (PPO, SAC, or Contextual LinUCB) on
SurfaceCodeReweightEnv to dynamically reweight PyMatching DEM graphs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from qecrl.envs.reweight_env import SurfaceCodeReweightEnv
from qecrl.agents.linucb_bandit import LinUCBBanditAgent


def train_bandit(distance=3, num_episodes=50, window_size=200, regime="all", save_path="results/models/linucb.npz"):
    print(f"\n[Training LinUCB Contextual Bandit on d={distance}, {num_episodes} episodes]")
    env = SurfaceCodeReweightEnv(distance=distance, window_size=window_size, max_windows=50, regime=regime)
    context_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = LinUCBBanditAgent(context_dim=context_dim, action_dim=action_dim, alpha=0.3)
    episode_rewards = []

    t0 = time.perf_counter()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0

        while not done:
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward)
            obs = next_obs
            ep_rew += reward
            done = terminated or truncated

        episode_rewards.append(ep_rew)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1:3d}/{num_episodes} | Mean Reward: {np.mean(episode_rewards[-10:]):.3f}")

    t1 = time.perf_counter()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"[Finished] Bandit trained in {t1 - t0:.2f}s. Saved to {save_path}")
    return agent


def train_sb3(algo="ppo", distance=3, total_timesteps=10000, window_size=200, regime="all", save_path="results/models"):
    print(f"\n[Training {algo.upper()} on SurfaceCodeReweightEnv (d={distance}, timesteps={total_timesteps})]")
    env = SurfaceCodeReweightEnv(
        distance=distance,
        window_size=window_size,
        max_windows=40,
        regime=regime,
    )

    save_dir = Path(save_path) / algo
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(net_arch=[128, 128])

    if algo.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            gamma=0.95,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="results/logs/ppo",
        )
    elif algo.lower() == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64,
            gamma=0.95,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="results/logs/sac",
        )
    else:
        raise ValueError(f"Unknown algo '{algo}'")

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    t1 = time.perf_counter()

    model_file = save_dir / f"{algo}_final_model.zip"
    model.save(str(model_file))
    print(f"\n[Finished] {algo.upper()} trained in {t1 - t0:.2f}s. Saved to: {model_file}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Meta-Controller for DEM Reweighting")
    parser.add_argument("--algo", type=str, default="linucb", choices=["linucb", "ppo", "sac"])
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=6000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--regime", type=str, default="all")

    args = parser.parse_args()

    if args.algo == "linucb":
        train_bandit(
            distance=args.distance,
            num_episodes=args.episodes,
            window_size=args.window_size,
            regime=args.regime,
        )
    else:
        train_sb3(
            algo=args.algo,
            distance=args.distance,
            total_timesteps=args.timesteps,
            window_size=args.window_size,
            regime=args.regime,
        )
