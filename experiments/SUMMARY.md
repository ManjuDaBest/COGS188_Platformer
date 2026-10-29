# Repeated platformer experiments

- Difficulty curriculum: easy → medium → hard; all policies are evaluated on the same seed set within each tier.
- Training runs: 2 seeds × 2 configurations for each of three learned agents.
- Q-learning budget: 400 episodes per run.
- Double DQN/PPO budget: 25,000 environment steps per run.
- Validation: 30 shared seeds.
- Held-out test: 50 shared seeds.
- Total wall time: 63.5 seconds.

## Held-out results

| Policy | Easy completion | Medium completion | Hard completion | Overall completion | Mean return | Best-attempt replays (easy / medium / hard) |
|---|---:|---:|---:|---:|---:|---|
| double_dqn | 74.0% | 28.0% | 16.0% | 39.3% | 2.59 | [seed 1221](./replays/double_dqn_easy_best_seed_1221.html) / [seed 1221](./replays/double_dqn_medium_best_seed_1221.html) / [seed 1220](./replays/double_dqn_hard_best_seed_1220.html) |
| ppo | 58.0% | 10.0% | 6.0% | 24.7% | 1.99 | [seed 1244](./replays/ppo_easy_best_seed_1244.html) / [seed 1203](./replays/ppo_medium_best_seed_1203.html) / [seed 1232](./replays/ppo_hard_best_seed_1232.html) |
| heuristic | 22.0% | 10.0% | 6.0% | 12.7% | -0.99 | [seed 1222](./replays/heuristic_easy_best_seed_1222.html) / [seed 1234](./replays/heuristic_medium_best_seed_1234.html) / [seed 1224](./replays/heuristic_hard_best_seed_1224.html) |
| q_learning | 12.0% | 0.0% | 0.0% | 4.0% | -4.47 | [seed 1235](./replays/q_learning_easy_best_seed_1235.html) / [seed 1235](./replays/q_learning_medium_best_seed_1235.html) / [seed 1200](./replays/q_learning_hard_best_seed_1200.html) |
| mcts | 4.0% | 0.0% | 0.0% | 1.3% | -5.93 | [seed 1222](./replays/mcts_easy_best_seed_1222.html) / [seed 1208](./replays/mcts_medium_best_seed_1208.html) / [seed 1230](./replays/mcts_hard_best_seed_1230.html) |
| random | 0.0% | 0.0% | 0.0% | 0.0% | -6.65 | [seed 1239](./replays/random_easy_best_seed_1239.html) / [seed 1228](./replays/random_medium_best_seed_1228.html) / [seed 1207](./replays/random_hard_best_seed_1207.html) |

Best-attempt replays are selected from the held-out episodes for inspection; they do not replace aggregate test metrics.

## Selected learned-agent configurations

| Agent | Configuration | Validation completion | Validation return | Training seed |
|---|---|---:|---:|---:|
| q_learning | `{"bins": 4, "epsilon_decay": 0.99, "learning_rate": 0.1, "name": "coarse_bins"}` | 1.7% | -5.49 | 17 |
| double_dqn | `{"learning_rate": 0.001, "name": "higher_lr", "target_update_interval": 500}` | 26.7% | 3.06 | 17 |
| ppo | `{"entropy_coefficient": 0.01, "learning_rate": 0.0003, "name": "standard_entropy"}` | 13.9% | 2.73 | 17 |

## Shared held-out level profiles

| Difficulty | Mean gap | Mean platform width | Moving platforms / level | Hazards / level | Enemies / level |
|---|---:|---:|---:|---:|---:|
| easy | 0.67 | 4.55 | 0.36 | 0.34 | 0.38 |
| medium | 1.15 | 3.81 | 1.66 | 1.34 | 1.16 |
| hard | 1.59 | 3.47 | 2.76 | 2.06 | 1.34 |

This is a CPU-sized preliminary experiment, not exhaustive tuning. MCTS uses the simulator during planning and is not directly comparable to learned agents by environment-step efficiency.
