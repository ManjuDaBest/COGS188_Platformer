# COGS188 Project template
Fork this repo.

Once you do it is is your repo. You are free to manage the repo as you see fit, edit this README, add data files, add scripts, etc. 

Also, you are free and encouraged to share this project after the course and to add it to your portfolio. Just be sure to fork it to your GitHub at the end of the quarter!

## Platformer reinforcement-learning implementation

The headless Gymnasium environment and experiment code live in `platformer/`.
It provides a seed-deterministic procedural level generator, a normalized
42-value observation, eight discrete movement actions, and optional
potential-based progress shaping.

Install the Python dependencies and run the test suite:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Train the agents and evaluate either a baseline or a saved checkpoint:

```bash
python3 -m platformer train --algorithm q --episodes 100 --output models/q.json
python3 -m platformer train --algorithm double-dqn --steps 10000 --output models/dqn.pt
python3 -m platformer train --algorithm ppo --steps 10000 --output models/ppo.pt
python3 -m platformer evaluate --algorithm heuristic --episodes 20 --seed-start 1000
python3 -m platformer evaluate --algorithm double-dqn --checkpoint models/dqn.pt --episodes 20
python3 -m platformer visualize --seed 11 --algorithm heuristic
python3 -m platformer visualize --seed 1200 --algorithm ppo --checkpoint models/ppo.pt --output replays/ppo.html
python3 -m platformer visualize --seed 1200 --difficulty hard --algorithm heuristic --output replays/hard.html
```

`visualize` writes a self-contained HTML replay. Open `platformer_episode.html`
in a browser to watch, pause, scrub, step through, and change playback speed.
The default heuristic run is useful before training a model; learned policies
can be visualized by supplying their checkpoint.

Use the assigned seed ranges consistently in experiments: `0-999` for
training, `1000-1199` for validation/model selection, and `1200-1499` for
final testing. The evaluation command reports episodic return, completion
rate, gem collection, and episode length as JSON. MCTS can be run with
`--algorithm mcts`; its simulator rollouts make it a planning reference, not
a sample-efficiency comparison. The CLI defaults to CPU. Scale the training
budget and evaluation episode count to the available compute; full results
should be reported over multiple training seeds, not inferred from a smoke
run.

## Algorithm-family model selection

The project is a **Markov decision process (MDP)**: the observation, eight
actions, transition dynamics, reward, discount, and terminal conditions define
the decision problem. An MDP is the common problem formulation, not an
algorithm competing with Q-learning or PPO.

| Family | Role in this project | Fit and caveat | Implementation status |
|---|---|---|---|
| Dynamic programming | Value/policy iteration with a known transition model | The simulator is deterministic for a level seed, but the physics state is continuous. Exact DP over the full state space is infeasible; discretization would produce an approximate, resolution-dependent oracle. | Analysis-only; not a current runnable baseline |
| Monte Carlo | Learn/evaluate from complete episodes; MCTS plans using sampled simulator rollouts | Episodic returns can have high variance. The implemented MCTS is a **planner with simulator access**, not Monte Carlo policy-control, and is reported separately from model-free learners. | MCTS is implemented; Monte Carlo control is not |
| Bandits | Contextual action-selection baseline | A bandit treats each action as an immediate reward choice and ignores how movement affects future states, making it a weak fit for platforming's long horizon. | Analysis-only |
| Local search | Optimize a short sequence of actions from the current state | Can exploit the forward simulator for short-horizon control, but open-loop action sequences are sensitive to later events and moving platforms. | Analysis-only; related planning reference is MCTS |
| Heuristic search | Use domain knowledge to guide actions/search toward the goal | Provides an interpretable benchmark. Its hand-written policy may fail on hazards or layouts that violate its assumptions. | Heuristic policy and heuristic-guided MCTS rollouts are implemented |
| Uninformed search | BFS/DFS over a state/action graph | Requires a finite graph. Continuous positions, velocities, and moving objects make the raw state space unsuitable; discretization introduces the same resolution trade-off as DP. | Analysis-only |
| Model-free RL | Learn from interaction without planning over a transition model | Directly matches the structured observation and allows procedural training. Tabular Q-learning is a coarse baseline; Double DQN and PPO are the main neural comparisons. | Q-learning, Double DQN, and PPO are implemented |

The initial empirical shortlist is Q-learning, Double DQN, and PPO; the
heuristic and random policies anchor baseline performance, and MCTS provides
a separate simulator-informed reference. This is a shortlist, **not a claim
that a best model has already been measured**. Select the winner by validation
completion rate across multiple training seeds, then use mean return and gem
collection as tie-breakers. Also report the generalization gap, steps to 80%
validation completion when attained, and CPU wall-clock time. Tune only on
seeds `1000-1199`, lock the configuration, and evaluate it once on held-out
seeds `1200-1499`. Compare learned methods at matched environment-step budgets;
report MCTS separately because it can query the simulator during planning and
its compute is not directly comparable to learner sample efficiency.

### Paired increasing-difficulty experiment

Run the bounded repeated comparison with:

```bash
python3 -m scripts.run_experiments
```

Experiments now use a shared easy → medium → hard curriculum. Every policy is
evaluated on identical 30 validation seeds and 50 held-out seeds **within each
difficulty tier**. The level generator increases platform gaps, narrows
platforms, and raises vertical variation and obstacle/moving-platform
probabilities across tiers. Training uses the same difficulty order and seed
bases for each learner; all evaluation reports retain the exact level profile
for auditing.

The current CPU-sized run trained two configurations with two model seeds for
each of Q-learning, Double DQN, and PPO. Each neural run used 25,000 steps
divided across the three difficulty stages; each Q-learning run used 400
episodes divided across the stages. On 150 held-out episodes per policy (50
per tier), Double DQN had the highest overall completion rate at 39.3%, with
74% easy, 28% medium, and 16% hard. PPO completed 24.7% overall; the heuristic
completed 12.7%. These are preliminary results from two training seeds, not
exhaustive tuning or a statistically conclusive benchmark.

The [experiment summary](./experiments/SUMMARY.md) links to a best-attempt
replay for **each policy at each difficulty**. Full per-level profiles and
metrics are in [the machine-readable report](./experiments/results.json).
Selected learner checkpoints are saved under `experiments/checkpoints/`.
