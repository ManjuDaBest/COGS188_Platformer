"""Run a reproducible, CPU-sized model comparison and save best-attempt replays."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from platformer.agents import (
    HeuristicPolicy,
    RandomPolicy,
    TabularQLearning,
    train_q_learning,
)
from platformer.deep_rl import DoubleDQN, PPO, train_double_dqn, train_ppo
from platformer.env import DIFFICULTY_CONFIGS, PlatformerEnv
from platformer.planning import MCTSPlanner
from platformer.visualize import create_episode_visualization


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments"
CHECKPOINTS = OUTPUT / "checkpoints"
REPLAYS = OUTPUT / "replays"
ENVIRONMENT = {"max_episode_steps": 300, "reward_shaping": True}
DIFFICULTIES = tuple(DIFFICULTY_CONFIGS)
TRAINING_SEED_BASES = {"easy": 10_000, "medium": 20_000, "hard": 30_000}
TRAIN_SEEDS = (7, 17)
VALIDATION_SEEDS = tuple(range(1000, 1030))
TEST_SEEDS = tuple(range(1200, 1250))
TRAINING_EPISODES = 400
TRAINING_STEPS = 25_000

CONFIGURATIONS: dict[str, list[dict[str, Any]]] = {
    "q_learning": [
        {"name": "coarse_bins", "bins": 4, "learning_rate": 0.1, "epsilon_decay": 0.99},
        {"name": "default_bins", "bins": 6, "learning_rate": 0.15, "epsilon_decay": 0.995},
    ],
    "double_dqn": [
        {"name": "higher_lr", "learning_rate": 1e-3, "target_update_interval": 500},
        {"name": "lower_lr", "learning_rate": 3e-4, "target_update_interval": 1000},
    ],
    "ppo": [
        {"name": "standard_entropy", "learning_rate": 3e-4, "entropy_coefficient": 0.01},
        {"name": "more_exploration", "learning_rate": 1e-4, "entropy_coefficient": 0.03},
    ],
}


def build_policy(name: str, seed: int, config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "mcts":
        return MCTSPlanner(simulations=8, rollout_horizon=8, seed=seed)
    if name == "q_learning":
        return TabularQLearning(
            bins=config.get("bins", 6),
            learning_rate=config.get("learning_rate", 0.15),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            seed=seed,
        )
    if name == "double_dqn":
        return DoubleDQN(
            PlatformerEnv.OBSERVATION_SIZE,
            8,
            learning_rate=config.get("learning_rate", 1e-3),
            exploration_steps=TRAINING_STEPS,
            replay_capacity=25_000,
            batch_size=64,
            target_update_interval=config.get("target_update_interval", 500),
            seed=seed,
        )
    if name == "ppo":
        return PPO(
            PlatformerEnv.OBSERVATION_SIZE,
            8,
            learning_rate=config.get("learning_rate", 3e-4),
            entropy_coefficient=config.get("entropy_coefficient", 0.01),
            rollout_steps=128,
            minibatch_size=64,
            epochs=4,
            seed=seed,
        )
    raise ValueError(f"unknown policy: {name}")


def train_candidate(
    name: str, policy: Any, model_seed: int, checkpoint: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    stage_runs = {}
    if name == "q_learning":
        episode_counts = {
            difficulty: TRAINING_EPISODES // len(DIFFICULTIES)
            for difficulty in DIFFICULTIES
        }
        for difficulty in DIFFICULTIES[: TRAINING_EPISODES % len(DIFFICULTIES)]:
            episode_counts[difficulty] += 1
        for difficulty in DIFFICULTIES:
            env = PlatformerEnv(**ENVIRONMENT, difficulty=difficulty)
            history = train_q_learning(
                env,
                policy,
                episodes=episode_counts[difficulty],
                seed_start=TRAINING_SEED_BASES[difficulty],
            )
            stage_runs[difficulty] = len(history)
            env.close()
    else:
        steps_per_stage = TRAINING_STEPS // len(DIFFICULTIES)
        for difficulty in DIFFICULTIES:
            env = PlatformerEnv(**ENVIRONMENT, difficulty=difficulty)
            train_function = (
                train_double_dqn if name == "double_dqn" else train_ppo
            )
            if name == "double_dqn":
                history = train_function(
                    env,
                    policy,
                    total_timesteps=steps_per_stage,
                    seed_start=TRAINING_SEED_BASES[difficulty],
                    learning_starts=500,
                    train_frequency=4,
                )
            else:
                history = train_function(
                    env,
                    policy,
                    total_timesteps=steps_per_stage,
                    seed_start=TRAINING_SEED_BASES[difficulty],
                )
            stage_runs[difficulty] = len(history)
            env.close()
    policy.save(checkpoint)
    return {
        "training_seconds": round(time.perf_counter() - started, 2),
        "episodes_completed_during_training": stage_runs,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }


def run_episode(
    name: str, policy: Any, seed: int, difficulty: str
) -> dict[str, Any]:
    env = PlatformerEnv(**ENVIRONMENT, difficulty=difficulty)
    observation, reset_info = env.reset(seed=seed)
    total_return = 0.0
    steps = 0
    while True:
        action = policy.act(env) if name == "mcts" else policy.act(
            observation, deterministic=True
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return {
        "seed": seed,
        "return": float(total_return),
        "completed": bool(info["completed"]),
        "gems_collected": int(info["gems_collected"]),
        "gems_available": int(info["gems_available"]),
        "steps": steps,
        "difficulty": difficulty,
        "level_profile": reset_info["level_profile"],
    }


def evaluate_seeds(
    name: str,
    policy_factory: Callable[[int], Any],
    seeds: tuple[int, ...],
    difficulty: str,
) -> dict[str, Any]:
    episodes = [
        run_episode(name, policy_factory(seed), seed, difficulty) for seed in seeds
    ]
    returns = [episode["return"] for episode in episodes]
    completion = [float(episode["completed"]) for episode in episodes]
    gem_fractions = [
        episode["gems_collected"] / max(episode["gems_available"], 1)
        for episode in episodes
    ]
    return {
        "episodes": episodes,
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.pstdev(returns),
        "completion_rate": statistics.fmean(completion),
        "mean_gem_fraction": statistics.fmean(gem_fractions),
        "mean_gems_collected": statistics.fmean(
            [episode["gems_collected"] for episode in episodes]
        ),
        "mean_steps": statistics.fmean([episode["steps"] for episode in episodes]),
    }


def evaluate_curriculum(
    name: str,
    policy_factory: Callable[[int], Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    by_difficulty = {
        difficulty: evaluate_seeds(name, policy_factory, seeds, difficulty)
        for difficulty in DIFFICULTIES
    }
    episodes = [
        episode
        for difficulty in DIFFICULTIES
        for episode in by_difficulty[difficulty]["episodes"]
    ]
    returns = [episode["return"] for episode in episodes]
    completions = [float(episode["completed"]) for episode in episodes]
    gem_fractions = [
        episode["gems_collected"] / max(episode["gems_available"], 1)
        for episode in episodes
    ]
    return {
        "by_difficulty": by_difficulty,
        "episodes": episodes,
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.pstdev(returns),
        "completion_rate": statistics.fmean(completions),
        "mean_gem_fraction": statistics.fmean(gem_fractions),
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        metrics["completion_rate"],
        metrics["mean_return"],
        metrics["mean_gem_fraction"],
    )


def best_episode(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        episodes,
        key=lambda item: (
            item["completed"],
            item["gems_collected"] / max(item["gems_available"], 1),
            item["return"],
            -item["steps"],
        ),
    )


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    REPLAYS.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    total_started = time.perf_counter()
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in CONFIGURATIONS
    }
    all_checkpoints: list[Path] = []

    for model_name, configurations in CONFIGURATIONS.items():
        for config in configurations:
            config_results = []
            for model_seed in TRAIN_SEEDS:
                policy = build_policy(model_name, model_seed, config)
                checkpoint = (
                    CHECKPOINTS
                    / f"{model_name}_{config['name']}_seed_{model_seed}"
                    / ("q.json" if model_name == "q_learning" else "agent.pt")
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                training = train_candidate(
                    model_name, policy, model_seed, checkpoint
                )
                all_checkpoints.append(checkpoint)
                validation = evaluate_curriculum(
                    model_name,
                    lambda seed, saved=checkpoint, cfg=config, name=model_name: load_policy(
                        name, saved, seed, cfg
                    ),
                    VALIDATION_SEEDS,
                )
                config_results.append(
                    {
                        "training_seed": model_seed,
                        "training": training,
                        "validation": validation,
                    }
                )
            aggregate = aggregate_validation(config_results)
            candidates[model_name].append(
                {
                    "configuration": config,
                    "runs": config_results,
                    "aggregate_validation": aggregate,
                }
            )
            print(
                f"{model_name} / {config['name']}: "
                f"validation completion={aggregate['completion_rate']:.1%}, "
                f"return={aggregate['mean_return']:.2f}"
            )

    selected: dict[str, dict[str, Any]] = {}
    for model_name, options in candidates.items():
        chosen = max(options, key=lambda row: selection_key(row["aggregate_validation"]))
        best_run = max(
            chosen["runs"], key=lambda row: selection_key(row["validation"])
        )
        checkpoint = Path(best_run["training"]["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        selected[model_name] = {
            "configuration": chosen["configuration"],
            "validation_summary": chosen["aggregate_validation"],
            "selected_training_seed": best_run["training_seed"],
            "selected_checkpoint": str(checkpoint.relative_to(ROOT)),
            "validation_run_metrics": best_run["validation"],
        }

    factories: dict[str, Callable[[int], Any]] = {
        "random": lambda seed: RandomPolicy(seed=seed),
        "heuristic": lambda seed: HeuristicPolicy(),
        "mcts": lambda seed: MCTSPlanner(
            simulations=8, rollout_horizon=8, seed=seed
        ),
    }
    for model_name, details in selected.items():
        checkpoint = ROOT / details["selected_checkpoint"]
        config = details["configuration"]
        factories[model_name] = (
            lambda seed, path=checkpoint, cfg=config, name=model_name: load_policy(
                name, path, seed, cfg
            )
        )

    test_results: dict[str, Any] = {}
    for name, factory in factories.items():
        metrics = evaluate_curriculum(name, factory, TEST_SEEDS)
        metrics["best_attempts"] = {}
        for difficulty, difficulty_metrics in metrics["by_difficulty"].items():
            chosen_episode = best_episode(difficulty_metrics["episodes"])
            replay_policy = factory(chosen_episode["seed"])
            replay_path = (
                REPLAYS
                / f"{name}_{difficulty}_best_seed_{chosen_episode['seed']}.html"
            )
            replay = create_episode_visualization(
                PlatformerEnv(**ENVIRONMENT, difficulty=difficulty),
                replay_policy,
                seed=chosen_episode["seed"],
                output_path=replay_path,
                algorithm=f"{name} · {difficulty}",
            )
            metrics["best_attempts"][difficulty] = {
                **chosen_episode,
                "replay": str(replay_path.relative_to(ROOT)),
                "replay_result": replay,
            }
        test_results[name] = metrics
        print(
            f"held-out {name}: "
            + ", ".join(
                f"{difficulty}={metrics['by_difficulty'][difficulty]['completion_rate']:.1%}"
                for difficulty in DIFFICULTIES
            )
            + f", overall={metrics['completion_rate']:.1%}"
        )

    level_profiles = {}
    for difficulty in DIFFICULTIES:
        profiles = [
            episode["level_profile"]
            for episode in test_results["random"]["by_difficulty"][difficulty][
                "episodes"
            ]
        ]
        level_profiles[difficulty] = {
            "difficulty_parameters": DIFFICULTY_CONFIGS[difficulty],
            "mean_gap": statistics.fmean(profile["mean_gap"] for profile in profiles),
            "mean_platform_width": statistics.fmean(
                profile["mean_platform_width"] for profile in profiles
            ),
            "mean_moving_platforms": statistics.fmean(
                profile["moving_platforms"] for profile in profiles
            ),
            "mean_hazards": statistics.fmean(
                profile["hazards"] for profile in profiles
            ),
            "mean_enemies": statistics.fmean(
                profile["enemies"] for profile in profiles
            ),
        }

    # Keep only the three checkpoints actually used for the reported held-out
    # comparison so the experiment output remains small and reproducible.
    for checkpoint in all_checkpoints:
        if checkpoint.exists() and all(
            checkpoint.resolve() != (ROOT / details["selected_checkpoint"]).resolve()
            for details in selected.values()
        ):
            checkpoint.unlink()
            parent = checkpoint.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

    report = {
        "experiment_design": {
            "training_runs_per_algorithm": len(TRAIN_SEEDS)
            * len(CONFIGURATIONS["q_learning"]),
            "configurations_per_algorithm": 2,
            "training_seeds": list(TRAIN_SEEDS),
            "q_learning_episodes_per_run": TRAINING_EPISODES,
            "neural_environment_steps_per_run": TRAINING_STEPS,
            "environment": ENVIRONMENT,
            "difficulty_schedule": {
                "order": list(DIFFICULTIES),
                "definitions": DIFFICULTY_CONFIGS,
                "training_seed_bases": TRAINING_SEED_BASES,
                "validation_seeds_per_difficulty": list(VALIDATION_SEEDS),
                "held_out_seeds_per_difficulty": list(TEST_SEEDS),
                "paired_levels": True,
            },
            "held_out_level_profile_summary": level_profiles,
            "validation_seeds_per_difficulty": list(VALIDATION_SEEDS),
            "held_out_test_seeds_per_difficulty": list(TEST_SEEDS),
            "selection_metric_order": [
                "completion rate",
                "mean episodic return",
                "mean fraction of gems collected",
            ],
            "best_replay_selection": (
                "One illustrative best episode per difficulty per policy, "
                "selected from held-out seeds by completion, gem fraction, "
                "return, then fewer steps. Replay selection does not change "
                "aggregate held-out metrics or model selection."
            ),
            "mcts_note": (
                "MCTS has access to the simulator during planning and is a "
                "planning reference, not a matched-sample learner."
            ),
        },
        "training_wall_seconds": round(time.perf_counter() - total_started, 2),
        "configurations": candidates,
        "selected_models": selected,
        "held_out_test": test_results,
        "caveats": [
            "This is a modest CPU experiment, not exhaustive hyperparameter tuning.",
            "Two training seeds and two configurations provide only a preliminary comparison.",
            "Each difficulty-specific best-attempt replay is selected from 50 test seeds; aggregate metrics use all 50.",
            "Bandits, dynamic programming, uninformed search, and general local/heuristic search have no separate executable policy in this implementation.",
        ],
    }
    report_path = OUTPUT / "results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary(report, OUTPUT / "SUMMARY.md")
    print(f"\nWrote {report_path.relative_to(ROOT)}")
    print(f"Total wall time: {report['training_wall_seconds']:.1f} seconds")


def aggregate_validation(runs: list[dict[str, Any]]) -> dict[str, float]:
    episode_groups = [run["validation"]["episodes"] for run in runs]
    pooled = [episode for group in episode_groups for episode in group]
    returns = [episode["return"] for episode in pooled]
    completions = [float(episode["completed"]) for episode in pooled]
    gem_fractions = [
        episode["gems_collected"] / max(episode["gems_available"], 1)
        for episode in pooled
    ]
    return {
        "completion_rate": statistics.fmean(completions),
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.pstdev(returns),
        "mean_gem_fraction": statistics.fmean(gem_fractions),
    }


def load_policy(
    name: str, checkpoint: Path, seed: int, config: dict[str, Any]
) -> Any:
    policy = build_policy(name, seed, config)
    policy.load(checkpoint)
    return policy


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Repeated platformer experiments",
        "",
        "- Difficulty curriculum: easy → medium → hard; all policies are evaluated on the same seed set within each tier.",
        f"- Training runs: {len(TRAIN_SEEDS)} seeds × 2 configurations for each of three learned agents.",
        f"- Q-learning budget: {TRAINING_EPISODES} episodes per run.",
        f"- Double DQN/PPO budget: {TRAINING_STEPS:,} environment steps per run.",
        f"- Validation: {len(VALIDATION_SEEDS)} shared seeds.",
        f"- Held-out test: {len(TEST_SEEDS)} shared seeds.",
        f"- Total wall time: {report['training_wall_seconds']:.1f} seconds.",
        "",
        "## Held-out results",
        "",
        "| Policy | Easy completion | Medium completion | Hard completion | Overall completion | Mean return | Best-attempt replays (easy / medium / hard) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in sorted(
        report["held_out_test"].items(),
        key=lambda item: (
            item[1]["completion_rate"],
            item[1]["mean_return"],
            item[1]["mean_gem_fraction"],
        ),
        reverse=True,
    ):
        replay_links = " / ".join(
            f"[seed {metrics['best_attempts'][difficulty]['seed']}]"
            f"(./{metrics['best_attempts'][difficulty]['replay'].removeprefix('experiments/')})"
            for difficulty in DIFFICULTIES
        )
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{metrics['by_difficulty'][difficulty]['completion_rate']:.1%}"
                for difficulty in DIFFICULTIES
            )
            + f" | {metrics['completion_rate']:.1%} | "
            + f"{metrics['mean_return']:.2f} | {replay_links} |"
        )
    lines.extend(
        [
            "",
            "Best-attempt replays are selected from the held-out episodes for inspection; "
            "they do not replace aggregate test metrics.",
            "",
            "## Selected learned-agent configurations",
            "",
            "| Agent | Configuration | Validation completion | Validation return | Training seed |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, model in report["selected_models"].items():
        validation = model["validation_summary"]
        lines.append(
            f"| {name} | `{json.dumps(model['configuration'], sort_keys=True)}` | "
            f"{validation['completion_rate']:.1%} | {validation['mean_return']:.2f} | "
            f"{model['selected_training_seed']} |"
        )
    profiles = report["experiment_design"]["held_out_level_profile_summary"]
    lines.extend(
        [
            "",
            "## Shared held-out level profiles",
            "",
            "| Difficulty | Mean gap | Mean platform width | Moving platforms / level | Hazards / level | Enemies / level |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for difficulty in DIFFICULTIES:
        profile = profiles[difficulty]
        lines.append(
            f"| {difficulty} | {profile['mean_gap']:.2f} | "
            f"{profile['mean_platform_width']:.2f} | "
            f"{profile['mean_moving_platforms']:.2f} | "
            f"{profile['mean_hazards']:.2f} | {profile['mean_enemies']:.2f} |"
        )
    lines.extend(
        [
            "",
            "This is a CPU-sized preliminary experiment, not exhaustive tuning. "
            "MCTS uses the simulator during planning and is not directly comparable "
            "to learned agents by environment-step efficiency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
