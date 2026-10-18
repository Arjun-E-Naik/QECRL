"""
QECRL — Contextual Bandit Meta-Controller (Fast LinUCB)
========================================================
High-performance LinUCB using Sherman-Morrison rank-1 updates to maintain
A_inv in O(d^2) time, eliminating matrix inversions during inference and training.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class LinUCBBanditAgent:
    """
    Optimized LinUCB Contextual Bandit with O(d^2) Sherman-Morrison updates.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        alpha: float = 1.2,
        ridge_lambda: float = 1.0,
        decay: float = 0.98,
    ):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.decay = decay

        # Maintain A_inv directly for each action dimension
        # Initial A = lambda * I -> A_inv = (1 / lambda) * I
        init_inv = np.eye(context_dim, dtype=np.float64) / ridge_lambda
        self.A_inv = np.array([init_inv.copy() for _ in range(action_dim)])
        self.b = np.zeros((action_dim, context_dim), dtype=np.float64)

    def predict(self, context: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Fast O(d^2) prediction for all action dimensions.
        """
        x_raw = context.astype(np.float64)
        norm = np.linalg.norm(x_raw)
        x = x_raw / max(norm, 1e-4)
        actions = np.zeros(self.action_dim, dtype=np.float32)

        for q in range(self.action_dim):
            A_inv_q = self.A_inv[q]
            theta_q = A_inv_q @ self.b[q]
            mean_pred = float(np.dot(theta_q, x))

            if not deterministic:
                # Variance: x^T A_inv x
                var = float(np.dot(x, A_inv_q @ x))
                ucb = self.alpha * np.sqrt(max(var, 1e-6))
                pred = mean_pred + ucb
            else:
                pred = mean_pred

            actions[q] = float(np.tanh(pred))

        return np.clip(actions, -1.0, 1.0)

    def update(self, context: np.ndarray, action: np.ndarray, reward: float):
        """
        Sherman-Morrison rank-1 update of A_inv in O(d^2) operations:
          (gamma * A + x x^T)^(-1) = (1/gamma) * [ A_inv - (A_inv x x^T A_inv) / (gamma + x^T A_inv x) ]
        """
        x_raw = context.astype(np.float64)
        norm = np.linalg.norm(x_raw)
        x = x_raw / max(norm, 1e-4)
        gamma = self.decay

        for q in range(self.action_dim):
            A_inv_q = self.A_inv[q]
            v = A_inv_q @ x  # (d,)
            denom = gamma + float(np.dot(x, v))

            # Rank-1 update
            self.A_inv[q] = (1.0 / gamma) * (A_inv_q - np.outer(v, v) / denom)

            # Update b
            self.b[q] = gamma * self.b[q] + (1.0 - gamma) * (reward * action[q] * x)

    def save(self, filepath: str):
        np.savez(filepath, A_inv=self.A_inv, b=self.b)

    def load(self, filepath: str):
        data = np.load(filepath)
        self.A_inv = data["A_inv"]
        self.b = data["b"]
