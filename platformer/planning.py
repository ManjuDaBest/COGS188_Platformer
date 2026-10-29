"""A small UCT Monte Carlo Tree Search reference planner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import numpy as np

from platformer.agents import HeuristicPolicy
from platformer.env import PlatformerEnv


@dataclass
class _Node:
    env: PlatformerEnv
    parent: _Node | None = None
    action: int | None = None
    edge_reward: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: list[_Node] = field(default_factory=list)
    untried_actions: list[int] = field(default_factory=list)
    done: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSPlanner:
    """Plan a short horizon using cloned simulator states and UCT selection."""

    def __init__(
        self,
        *,
        simulations: int = 96,
        rollout_horizon: int = 24,
        exploration_constant: float = 1.4,
        discount: float = 0.99,
        seed: int | None = None,
    ) -> None:
        if simulations <= 0 or rollout_horizon <= 0:
            raise ValueError("simulations and rollout_horizon must be positive")
        self.simulations = simulations
        self.rollout_horizon = rollout_horizon
        self.exploration_constant = exploration_constant
        self.discount = discount
        self.rng = np.random.default_rng(seed)
        self.rollout_policy = HeuristicPolicy()

    def act(self, env: PlatformerEnv) -> int:
        if env._episode_over:
            raise RuntimeError("cannot plan from a finished episode")
        root = _Node(
            env=deepcopy(env),
            untried_actions=list(range(env.action_space.n)),
        )
        for _ in range(self.simulations):
            node = root
            simulation_env = deepcopy(root.env)
            path = [root]
            depth = 0
            while not node.untried_actions and node.children and not node.done:
                node = self._select_child(node)
                observation, reward, terminated, truncated, _ = simulation_env.step(node.action)
                node.edge_reward = reward
                node.done = terminated or truncated
                depth += 1
                path.append(node)
            if node.untried_actions and not node.done:
                action_index = int(self.rng.integers(len(node.untried_actions)))
                action = node.untried_actions.pop(action_index)
                observation, reward, terminated, truncated, _ = simulation_env.step(action)
                child = _Node(
                    env=deepcopy(simulation_env),
                    parent=node,
                    action=action,
                    edge_reward=reward,
                    untried_actions=list(range(simulation_env.action_space.n)),
                    done=terminated or truncated,
                )
                node.children.append(child)
                node = child
                depth += 1
                path.append(node)
            rollout_value = self._rollout(simulation_env, self.rollout_horizon - depth)
            value = rollout_value
            for visited in reversed(path):
                if visited is not root:
                    value = visited.edge_reward + self.discount * value
                visited.visits += 1
                visited.value_sum += value
        if not root.children:
            return self.rollout_policy.act(root.env._observation())
        action = max(root.children, key=lambda child: child.visits).action
        if action is None:
            raise RuntimeError("MCTS produced a root child without an action")
        return action

    def _select_child(self, node: _Node) -> _Node:
        log_parent = math.log(max(node.visits, 1))
        return max(
            node.children,
            key=lambda child: child.mean_value
            + self.exploration_constant * math.sqrt(log_parent / max(child.visits, 1)),
        )

    def _rollout(self, env: PlatformerEnv, steps: int) -> float:
        total, multiplier = 0.0, 1.0
        if env._episode_over:
            return total
        for _ in range(max(0, steps)):
            observation = env._observation()
            action = self.rollout_policy.act(observation)
            _, reward, terminated, truncated, _ = env.step(action)
            total += multiplier * reward
            multiplier *= self.discount
            if terminated or truncated:
                break
        return total
