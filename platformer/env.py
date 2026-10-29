"""A deterministic, procedurally generated, headless platformer environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DIFFICULTY_CONFIGS = {
    "easy": {
        "gap": (0.45, 0.9),
        "width": (4.0, 5.0),
        "height_change": 0.3,
        "moving_platform_probability": 0.05,
        "hazard_probability": 0.05,
        "enemy_probability": 0.05,
    },
    "medium": {
        "gap": (0.7, 1.6),
        "width": (3.0, 4.3),
        "height_change": 0.7,
        "moving_platform_probability": 0.22,
        "hazard_probability": 0.16,
        "enemy_probability": 0.2,
    },
    "hard": {
        "gap": (1.2, 2.0),
        "width": (2.7, 3.8),
        "height_change": 0.9,
        "moving_platform_probability": 0.4,
        "hazard_probability": 0.28,
        "enemy_probability": 0.28,
    },
}


@dataclass
class Platform:
    left: float
    width: float
    y: float
    moving: bool = False
    amplitude: float = 0.0
    phase: float = 0.0
    speed: float = 0.0

    @property
    def right(self) -> float:
        return self.left + self.width

    def height_at(self, time: float) -> float:
        if not self.moving:
            return self.y
        return self.y + self.amplitude * math.sin(self.phase + time * self.speed)


@dataclass
class Hazard:
    x: float
    y: float


@dataclass
class Enemy:
    x: float
    y: float
    origin_x: float
    amplitude: float
    phase: float
    speed: float

    def position_at(self, time: float) -> tuple[float, float]:
        return self.origin_x + self.amplitude * math.sin(self.phase + time * self.speed), self.y


@dataclass
class Gem:
    x: float
    y: float
    collected: bool = False


class PlatformerEnv(gym.Env[np.ndarray, int]):
    """Simple 2D platformer with a fixed-width state vector and eight actions.

    Actions are: no-op, left, right, jump, left+jump, right+jump, glide,
    and right+glide. Level geometry is a pure function of the reset seed.
    """

    metadata = {"render_modes": []}

    ACTIONS = (
        (0, False, False),
        (-1, False, False),
        (1, False, False),
        (0, True, False),
        (-1, True, False),
        (1, True, False),
        (0, False, True),
        (1, False, True),
    )
    PLATFORM_COUNT = 4
    OBJECT_COUNT = 2
    OBSERVATION_SIZE = 6 + PLATFORM_COUNT * 4 + OBJECT_COUNT * 3 * 3 + 2

    def __init__(
        self,
        *,
        level_platforms: int = 9,
        max_episode_steps: int = 500,
        reward_shaping: bool = True,
        difficulty: str = "medium",
    ) -> None:
        super().__init__()
        if level_platforms < 3:
            raise ValueError("level_platforms must be at least 3")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if difficulty not in DIFFICULTY_CONFIGS:
            raise ValueError(
                f"difficulty must be one of {', '.join(DIFFICULTY_CONFIGS)}"
            )
        self.level_platforms = level_platforms
        self.max_episode_steps = max_episode_steps
        self.reward_shaping = reward_shaping
        self.difficulty = difficulty
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBSERVATION_SIZE,), dtype=np.float32
        )
        self.platforms: list[Platform] = []
        self.hazards: list[Hazard] = []
        self.enemies: list[Enemy] = []
        self.gems: list[Gem] = []
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.glide_available = True
        self.elapsed_steps = 0
        self._level_time = 0.0
        self._rng = np.random.default_rng()
        self._episode_over = True
        self._gems_collected = 0
        self._last_seed: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "difficulty" in options:
            requested_difficulty = options["difficulty"]
            if requested_difficulty not in DIFFICULTY_CONFIGS:
                raise ValueError(
                    f"difficulty must be one of {', '.join(DIFFICULTY_CONFIGS)}"
                )
            self.difficulty = requested_difficulty
        if seed is not None:
            self._last_seed = int(seed)
        else:
            self._last_seed = int(self.np_random.integers(0, 2**32 - 1))
        self._rng = np.random.default_rng(self._last_seed)
        self._generate_level()
        self.x = self.platforms[0].left + 1.0
        self.y = self.platforms[0].height_at(0.0) + 0.5
        self.vx = self.vy = 0.0
        self.on_ground = True
        self.glide_available = True
        self.elapsed_steps = 0
        self._level_time = 0.0
        self._episode_over = False
        self._gems_collected = 0
        return self._observation(), {
            "seed": self._last_seed,
            "difficulty": self.difficulty,
            "level_profile": self.level_profile(),
        }

    def level_profile(self) -> dict[str, float | int | str]:
        gaps = [
            self.platforms[index].left - self.platforms[index - 1].right
            for index in range(1, len(self.platforms))
        ]
        widths = [platform.width for platform in self.platforms]
        return {
            "difficulty": self.difficulty,
            "mean_gap": float(np.mean(gaps)),
            "max_gap": float(max(gaps)),
            "mean_platform_width": float(np.mean(widths)),
            "moving_platforms": sum(platform.moving for platform in self.platforms),
            "hazards": len(self.hazards),
            "enemies": len(self.enemies),
        }

    def _generate_level(self) -> None:
        self.platforms = []
        self.hazards = []
        self.enemies = []
        self.gems = []
        left, width, height = -2.0, 5.0, 0.0
        config = DIFFICULTY_CONFIGS[self.difficulty]
        for index in range(self.level_platforms):
            if index:
                gap = self._rng.uniform(*config["gap"])
                left += width + float(gap)
                width = float(self._rng.uniform(*config["width"]))
                height = float(
                    np.clip(
                        height
                        + self._rng.uniform(
                            -config["height_change"], config["height_change"]
                        ),
                        -1.5,
                        1.5,
                    )
                )
            moving = (
                index > 0
                and index < self.level_platforms - 1
                and self._rng.random() < config["moving_platform_probability"]
            )
            platform = Platform(
                left=left,
                width=width,
                y=height,
                moving=moving,
                amplitude=float(self._rng.uniform(0.15, 0.45)) if moving else 0.0,
                phase=float(self._rng.uniform(0.0, 2.0 * math.pi)),
                speed=float(self._rng.uniform(0.8, 1.5)) if moving else 0.0,
            )
            self.platforms.append(platform)
            if 0 < index < self.level_platforms - 1:
                if self._rng.random() < config["hazard_probability"]:
                    hx = left + float(self._rng.uniform(0.8, max(0.81, width - 0.5)))
                    self.hazards.append(Hazard(hx, height + 0.35))
                elif self._rng.random() < config["enemy_probability"]:
                    ex = left + width / 2.0
                    self.enemies.append(
                        Enemy(
                            ex,
                            height + 0.45,
                            ex,
                            float(self._rng.uniform(0.3, min(0.8, width / 4))),
                            float(self._rng.uniform(0.0, 2.0 * math.pi)),
                            float(self._rng.uniform(1.0, 2.0)),
                        )
                    )
            self.gems.append(Gem(left + width / 2.0, height + 1.1))
        last = self.platforms[-1]
        self.goal_x = last.right - 0.65
        self.goal_y = last.y

    def _observation(self) -> np.ndarray:
        values: list[float] = [
            self.x / 50.0,
            self.y / 10.0,
            self.vx / 6.0,
            self.vy / 10.0,
            float(self.on_ground),
            float(self.glide_available),
        ]
        platforms = sorted(self.platforms, key=lambda p: abs((p.left + p.width / 2) - self.x))
        for index in range(self.PLATFORM_COUNT):
            if index < len(platforms):
                p = platforms[index]
                values.extend(
                    [
                        ((p.left + p.width / 2) - self.x) / 20.0,
                        (p.height_at(self._level_time) - self.y) / 10.0,
                        p.width / 5.0,
                        float(p.moving),
                    ]
                )
            else:
                values.extend([0.0] * 4)
        for objects in (self.hazards, self.enemies, [g for g in self.gems if not g.collected]):
            if objects and isinstance(objects[0], Hazard):
                positions = [(obj.x, obj.y, 1.0) for obj in objects]  # type: ignore[attr-defined]
            elif objects and isinstance(objects[0], Enemy):
                positions = [
                    (*obj.position_at(self._level_time), 1.0) for obj in objects  # type: ignore[attr-defined]
                ]
            else:
                positions = [(obj.x, obj.y, 1.0) for obj in objects]  # type: ignore[attr-defined]
            positions.sort(key=lambda pos: abs(pos[0] - self.x))
            for index in range(self.OBJECT_COUNT):
                if index < len(positions):
                    ox, oy, kind = positions[index]
                    values.extend([(ox - self.x) / 20.0, (oy - self.y) / 10.0, kind])
                else:
                    values.extend([0.0, 0.0, 0.0])
        values.extend([(self.goal_x - self.x) / 50.0, (self.goal_y - self.y) / 10.0])
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_over:
            raise RuntimeError("step() called after episode end; call reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.action_space.n - 1}]")
        direction, jump, glide = self.ACTIONS[action]
        previous_potential = self._potential()
        previous_on_ground = self.on_ground
        dt = 0.1
        if direction:
            self.vx = float(np.clip(self.vx + direction * 2.2, -5.0, 5.0))
        else:
            self.vx *= 0.78
        if jump and previous_on_ground:
            self.vy = 8.0
            self.on_ground = False
        if glide and not self.on_ground and self.glide_available and self.vy < 0.0:
            self.vy = max(self.vy, -1.6)
            self.glide_available = False
        gravity = 5.0 if glide and not self.on_ground and self.glide_available else 18.0
        if glide and not self.on_ground and not self.glide_available:
            gravity = 5.0
        self.vy -= gravity * dt
        old_y = self.y
        self.x += self.vx * dt
        self.y += self.vy * dt
        self._level_time += dt
        self.elapsed_steps += 1
        self.on_ground = False
        if self.vy <= 0.0:
            for platform in self.platforms:
                top = platform.height_at(self._level_time)
                if (
                    platform.left - 0.3 <= self.x <= platform.right + 0.3
                    and old_y - 0.5 >= top
                    and self.y - 0.5 <= top
                ):
                    self.y = top + 0.5
                    self.vy = 0.0
                    self.on_ground = True
                    self.glide_available = True
                    break
        died = self.y < -8.0 or any(
            abs(self.x - hazard.x) < 0.48 and abs(self.y - hazard.y) < 0.65
            for hazard in self.hazards
        )
        if not died:
            for enemy in self.enemies:
                ex, ey = enemy.position_at(self._level_time)
                if abs(self.x - ex) < 0.65 and abs(self.y - ey) < 0.75:
                    died = True
                    break
        gem_reward = 0.0
        for gem in self.gems:
            if not gem.collected and math.hypot(self.x - gem.x, self.y - gem.y) < 0.8:
                gem.collected = True
                self._gems_collected += 1
                gem_reward += 1.0
        completed = self.x >= self.goal_x and abs(self.y - self.goal_y) < 3.5
        terminated = died or completed
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        reward = gem_reward - 0.01
        if self.reward_shaping:
            next_potential = 0.0 if terminated else self._potential()
            reward += 2.0 * (0.99 * next_potential - previous_potential)
        if completed:
            reward += 10.0
        elif died:
            reward -= 10.0
        self._episode_over = terminated or truncated
        info = {
            "completed": completed,
            "died": died,
            "gems_collected": self._gems_collected,
            "gems_available": len(self.gems),
            "level_seed": self._last_seed,
            "difficulty": self.difficulty,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _potential(self) -> float:
        return -abs(self.goal_x - self.x) / max(self.goal_x - self.platforms[0].left, 1.0)

    def close(self) -> None:
        return None
