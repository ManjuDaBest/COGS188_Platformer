"""Command-line training and evaluation entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from platformer.agents import (
    HeuristicPolicy,
    RandomPolicy,
    TabularQLearning,
    train_q_learning,
)
from platformer.deep_rl import DoubleDQN, PPO, train_double_dqn, train_ppo
from platformer.env import PlatformerEnv
from platformer.evaluation import evaluate_policy
from platformer.planning import MCTSPlanner
from platformer.visualize import create_episode_visualization


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _add_environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=500)


def train(args: argparse.Namespace) -> None:
    env = PlatformerEnv(
        max_episode_steps=args.max_episode_steps,
        reward_shaping=not args.no_shaping,
    )
    if args.algorithm == "q":
        agent = TabularQLearning(bins=args.bins, seed=args.seed)
        history = train_q_learning(env, agent, episodes=args.episodes, seed_start=args.seed)
    elif args.algorithm == "double-dqn":
        agent = DoubleDQN(
            env.observation_space.shape[0],
            env.action_space.n,
            seed=args.seed,
            exploration_steps=max(args.steps, 1),
        )
        history = train_double_dqn(
            env,
            agent,
            total_timesteps=args.steps,
            seed_start=args.seed,
            learning_starts=min(1_000, max(args.steps // 10, 1)),
        )
    else:
        agent = PPO(
            env.observation_space.shape[0],
            env.action_space.n,
            seed=args.seed,
        )
        history = train_ppo(env, agent, total_timesteps=args.steps, seed_start=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    history_path = args.output.with_name(f"{args.output.stem}_training.json")
    _write_json(history_path, {"algorithm": args.algorithm, "history": history})
    print(f"Saved {args.algorithm} checkpoint to {args.output}")
    env.close()


def _load_policy(args: argparse.Namespace, env: PlatformerEnv) -> Any:
    if args.algorithm == "random":
        return RandomPolicy(args.seed)
    if args.algorithm == "heuristic":
        return HeuristicPolicy()
    if args.algorithm == "mcts":
        return MCTSPlanner(simulations=args.simulations, seed=args.seed)
    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required for {args.algorithm}")
    if args.algorithm == "q":
        policy = TabularQLearning(bins=args.bins, seed=args.seed)
    elif args.algorithm == "double-dqn":
        policy = DoubleDQN(env.observation_space.shape[0], env.action_space.n, seed=args.seed)
    else:
        policy = PPO(env.observation_space.shape[0], env.action_space.n, seed=args.seed)
    policy.load(args.checkpoint)
    return policy


def evaluate(args: argparse.Namespace) -> None:
    env = PlatformerEnv(
        max_episode_steps=args.max_episode_steps,
        reward_shaping=not args.no_shaping,
        difficulty=args.difficulty,
    )
    policy = _load_policy(args, env)
    summary = evaluate_policy(env, policy, seeds=range(args.seed_start, args.seed_start + args.episodes))
    summary["algorithm"] = args.algorithm
    print(json.dumps(summary, indent=2))
    if args.output:
        _write_json(args.output, summary)
    env.close()


def visualize(args: argparse.Namespace) -> None:
    env = PlatformerEnv(
        max_episode_steps=args.max_episode_steps,
        reward_shaping=not args.no_shaping,
        difficulty=args.difficulty,
    )
    policy = _load_policy(args, env)
    summary = create_episode_visualization(
        env,
        policy,
        seed=args.seed,
        output_path=args.output,
        algorithm=args.algorithm,
    )
    print(json.dumps({**summary, "visualization": str(args.output)}, indent=2))
    env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="platformer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train a learning agent")
    train_parser.add_argument("--algorithm", choices=("q", "double-dqn", "ppo"), required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--episodes", type=int, default=100)
    train_parser.add_argument("--bins", type=int, default=6)
    _add_environment_options(train_parser)
    train_parser.set_defaults(func=train)
    eval_parser = subparsers.add_parser("evaluate", help="evaluate a policy on fixed level seeds")
    eval_parser.add_argument(
        "--algorithm",
        choices=("random", "heuristic", "mcts", "q", "double-dqn", "ppo"),
        required=True,
    )
    eval_parser.add_argument("--checkpoint", type=Path)
    eval_parser.add_argument("--episodes", type=int, default=20)
    eval_parser.add_argument("--seed-start", type=int, default=1_000)
    eval_parser.add_argument("--bins", type=int, default=6)
    eval_parser.add_argument("--simulations", type=int, default=96)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    _add_environment_options(eval_parser)
    eval_parser.set_defaults(func=evaluate)
    visualize_parser = subparsers.add_parser(
        "visualize", help="save a playable HTML replay of one agent episode"
    )
    visualize_parser.add_argument(
        "--algorithm",
        choices=("random", "heuristic", "mcts", "q", "double-dqn", "ppo"),
        default="heuristic",
    )
    visualize_parser.add_argument("--checkpoint", type=Path)
    visualize_parser.add_argument("--output", type=Path, default=Path("platformer_episode.html"))
    visualize_parser.add_argument("--bins", type=int, default=6)
    visualize_parser.add_argument("--simulations", type=int, default=96)
    visualize_parser.add_argument(
        "--difficulty", choices=("easy", "medium", "hard"), default="medium"
    )
    _add_environment_options(visualize_parser)
    visualize_parser.set_defaults(func=visualize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
