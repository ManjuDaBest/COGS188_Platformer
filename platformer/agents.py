"""Simple baseline and tabular reinforcement-learning policies."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int: ...


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self.rng.integers(0, 8))


class HeuristicPolicy:
    """Run toward the goal, jumping near gaps and visible threats."""

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        del deterministic
        on_ground = observation[4] > 0.5
        glide_available = observation[5] > 0.5
        goal_dx = float(observation[40])
        right_edge_dx = None
        next_platform_dx = None
        for i in range(4):
            offset = 6 + 4 * i
            dx, width = float(observation[offset]), float(observation[offset + 2])
            if width <= 0.0:
                continue
            edge = dx * 20.0 + width * 2.5
            if edge >= -0.5 and (right_edge_dx is None or edge < right_edge_dx):
                right_edge_dx = edge
            if dx > 0 and (next_platform_dx is None or dx < next_platform_dx):
                next_platform_dx = dx
        should_jump = on_ground and right_edge_dx is not None and right_edge_dx < 1.8
        for start in (22, 28):
            for i in range(2):
                dx = float(observation[start + i * 3])
                kind = float(observation[start + i * 3 + 2])
                if kind > 0 and 0.0 < dx < 0.07:
                    should_jump = on_ground
        if should_jump and (next_platform_dx is None or next_platform_dx < 0.29):
            return 5
        if not on_ground and observation[3] < -0.1 and glide_available:
            return 7 if goal_dx > 0 else 6
        return 2 if goal_dx > 0.01 else 1


class TabularQLearning:
    """Sparse ε-greedy Q-learning over a coarse discretization of state."""

    FEATURE_INDICES = (2, 3, 4, 5, 6, 7, 22, 23, 40)
    BOUNDS = ((-1, 1), (-1, 1), (0, 1), (0, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1))

    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        discount: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        bins: int = 6,
        seed: int | None = None,
    ) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(8, dtype=np.float32)
        )

    def state_key(self, observation: np.ndarray) -> tuple[int, ...]:
        result = []
        for index, (low, high) in zip(self.FEATURE_INDICES, self.BOUNDS):
            scaled = (float(np.clip(observation[index], low, high)) - low) / (high - low)
            result.append(min(int(scaled * self.bins), self.bins - 1))
        return tuple(result)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        key = self.state_key(observation)
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 8))
        return int(np.argmax(self.q_values[key]))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        current = self.q_values[self.state_key(observation)]
        next_value = 0.0 if terminated or truncated else float(
            np.max(self.q_values[self.state_key(next_observation)])
        )
        target = reward + self.discount * next_value
        current[action] += self.learning_rate * (target - current[action])

    def finish_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        rows = [
            {"state": list(state), "values": values.tolist()}
            for state, values in self.q_values.items()
        ]
        Path(path).write_text(json.dumps({"bins": self.bins, "q_values": rows}))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        if payload["bins"] != self.bins:
            raise ValueError("checkpoint bin count does not match this policy")
        self.q_values.clear()
        for row in payload["q_values"]:
            self.q_values[tuple(row["state"])] = np.asarray(row["values"], dtype=np.float32)


def train_q_learning(
    env: object,
    agent: TabularQLearning,
    *,
    episodes: int,
    seed_start: int = 0,
) -> list[dict[str, float]]:
    """Train an agent against deterministic level seeds and return episode logs."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    history = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed_start + episode)  # type: ignore[attr-defined]
        total_reward = 0.0
        completed = False
        gems = 0
        while True:
            action = agent.act(observation, deterministic=False)
            next_observation, reward, terminated, truncated, info = env.step(action)  # type: ignore[attr-defined]
            agent.update(observation, action, reward, next_observation, terminated, truncated)
            observation = next_observation
            total_reward += reward
            gems = int(info["gems_collected"])
            if terminated or truncated:
                completed = bool(info["completed"])
                break
        agent.finish_episode()
        history.append(
            {"episode": float(episode), "return": total_reward, "completed": float(completed), "gems": float(gems)}
        )
    return history
