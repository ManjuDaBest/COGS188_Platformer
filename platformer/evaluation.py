"""Seeded evaluation and summary metrics for platformer policies."""

from __future__ import annotations

from typing import Any
import numpy as np

from platformer.env import PlatformerEnv
from platformer.planning import MCTSPlanner


def evaluate_policy(
    env: PlatformerEnv,
    policy: Any,
    *,
    seeds: list[int] | range,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    episodes = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        total_return = 0.0
        steps = 0
        while True:
            if isinstance(policy, MCTSPlanner):
                action = policy.act(env)
            else:
                action = policy.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            steps += 1
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": int(seed),
                "return": total_return,
                "completed": bool(info["completed"]),
                "gems_collected": int(info["gems_collected"]),
                "gems_available": int(info["gems_available"]),
                "steps": steps,
            }
        )
    returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
    completions = np.asarray([episode["completed"] for episode in episodes], dtype=np.float64)
    gem_fractions = np.asarray(
        [
            episode["gems_collected"] / max(episode["gems_available"], 1)
            for episode in episodes
        ],
        dtype=np.float64,
    )
    return {
        "episodes": episodes,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "completion_rate": float(completions.mean()),
        "mean_gem_fraction": float(gem_fractions.mean()),
        "mean_gems_collected": float(
            np.mean([episode["gems_collected"] for episode in episodes])
        ),
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
    }


def steps_to_threshold(
    evaluations: list[tuple[int, float]], threshold: float = 0.8
) -> int | None:
    """Return first training step at which validation completion meets threshold."""
    for steps, completion_rate in sorted(evaluations):
        if completion_rate >= threshold:
            return steps
    return None


def generalization_gap(train_completion: float, test_completion: float) -> float:
    return float(train_completion - test_completion)
