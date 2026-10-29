"""Compact PyTorch implementations of Double DQN and PPO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(observation)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        return self.rng.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


class DoubleDQN:
    """Double DQN with replay, a target network, and linear epsilon decay."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        *,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        exploration_steps: int = 50_000,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_interval: int = 500,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if observation_size <= 0 or action_count <= 0:
            raise ValueError("observation_size and action_count must be positive")
        if batch_size <= 0 or replay_capacity < batch_size or target_update_interval <= 0:
            raise ValueError("batch_size, replay_capacity, and target_update_interval are invalid")
        _seed_torch(seed)
        self.action_count = action_count
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = max(1, exploration_steps)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)
        self.online = QNetwork(observation_size, action_count).to(self.device)
        self.target = QNetwork(observation_size, action_count).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def epsilon(self) -> float:
        fraction = min(self.steps / self.exploration_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            return int(self.online(tensor).argmax().item())

    def observe(self, transition: Transition) -> float | None:
        self.replay.add(transition)
        self.steps += 1
        return None

    def learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        obs = torch.as_tensor(np.stack([t.observation for t in batch]), device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([t.next_observation for t in batch]), device=self.device)
        done = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        q_value = self.online(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            best_next_action = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, best_next_action).squeeze(1)
            target = rewards + self.discount * (1.0 - done) * next_q
        loss = nn.functional.smooth_l1_loss(q_value, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {"online": self.online.state_dict(), "target": self.target.state_dict(), "steps": self.steps},
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.steps = int(payload["steps"])


def train_double_dqn(
    env: Any,
    agent: DoubleDQN,
    *,
    total_timesteps: int,
    seed_start: int = 0,
    learning_starts: int = 1_000,
    train_frequency: int = 4,
) -> list[dict[str, float]]:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if learning_starts < 0 or train_frequency <= 0:
        raise ValueError("learning_starts must be non-negative and train_frequency positive")
    observation, _ = env.reset(seed=seed_start)
    episode_return = 0.0
    episode = 0
    history: list[dict[str, float]] = []
    for step in range(1, total_timesteps + 1):
        action = agent.act(observation, deterministic=False)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.observe(Transition(observation, action, reward, next_observation, terminated))
        if step >= learning_starts and step % train_frequency == 0:
            agent.learn()
        if step % agent.target_update_interval == 0:
            agent.target.load_state_dict(agent.online.state_dict())
        observation = next_observation
        episode_return += reward
        if done:
            history.append(
                {
                    "episode": float(episode),
                    "step": float(step),
                    "return": episode_return,
                    "completed": float(info["completed"]),
                    "gems": float(info["gems_collected"]),
                }
            )
            episode += 1
            episode_return = 0.0
            observation, _ = env.reset(seed=seed_start + episode)
    return history


class PPO:
    """Clipped actor-critic PPO with generalized advantage estimation."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        *,
        learning_rate: float = 3e-4,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        rollout_steps: int = 128,
        epochs: int = 4,
        minibatch_size: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if rollout_steps <= 0 or epochs <= 0 or minibatch_size <= 0:
            raise ValueError("rollout_steps, epochs, and minibatch_size must be positive")
        if not 0.0 < clip_range < 1.0:
            raise ValueError("clip_range must be between 0 and 1")
        _seed_torch(seed)
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)
        self.model = ActorCritic(observation_size, action_count).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            logits, _ = self.model(tensor)
            if deterministic:
                return int(logits.argmax().item())
            return int(Categorical(logits=logits).sample().item())

    def value(self, observation: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            return float(self.model(tensor)[1].item())

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> float:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        total_loss = 0.0
        samples = len(obs)
        for _ in range(self.epochs):
            indices = torch.randperm(samples, device=self.device)
            for start in range(0, samples, self.minibatch_size):
                batch = indices[start : start + self.minibatch_size]
                logits, values = self.model(obs[batch])
                distribution = Categorical(logits=logits)
                log_prob = distribution.log_prob(acts[batch])
                ratio = torch.exp(log_prob - old_log[batch])
                unclipped = ratio * adv[batch]
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, ret[batch])
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += float(loss.item()) * len(batch)
        return total_loss / (samples * self.epochs)

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def train_ppo(
    env: Any,
    agent: PPO,
    *,
    total_timesteps: int,
    seed_start: int = 0,
) -> list[dict[str, float]]:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    observation, _ = env.reset(seed=seed_start)
    history: list[dict[str, float]] = []
    episode_return, episode, episodes_seeded = 0.0, 0, 0
    steps = 0
    while steps < total_timesteps:
        count = min(agent.rollout_steps, total_timesteps - steps)
        observations, actions, log_probs = [], [], []
        rewards, values, next_values, episode_ends = [], [], [], []
        for _ in range(count):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                logits, value = agent.model(obs_tensor)
                distribution = Categorical(logits=logits)
                action_tensor = distribution.sample()
                log_prob = distribution.log_prob(action_tensor)
            action = int(action_tensor.item())
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observations.append(observation)
            actions.append(action)
            log_probs.append(float(log_prob.item()))
            rewards.append(float(reward))
            values.append(float(value.item()))
            next_values.append(0.0 if terminated else agent.value(next_observation))
            episode_ends.append(float(done))
            observation = next_observation
            episode_return += reward
            steps += 1
            if done:
                history.append(
                    {
                        "episode": float(episode),
                        "step": float(steps),
                        "return": episode_return,
                        "completed": float(info["completed"]),
                        "gems": float(info["gems_collected"]),
                    }
                )
                episode += 1
                episodes_seeded += 1
                episode_return = 0.0
                observation, _ = env.reset(seed=seed_start + episodes_seeded)
        advantages = np.zeros(count, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(count)):
            continuation = 1.0 - episode_ends[index]
            delta = rewards[index] + agent.discount * next_values[index] - values[index]
            gae = delta + agent.discount * agent.gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        agent.update(
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(log_probs, dtype=np.float32),
            returns,
            advantages,
        )
    return history
