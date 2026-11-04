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

Project Proposal: Improving Hard-Level Generalization in a Procedural Platformer
Team
Team members: Adi Majumdar, Hendrix, Samuel.
Proposed project type: AI solve-a-problem project; reinforcement learning.
Abstract
This project will test whether an adaptive difficulty curriculum helps reinforcement-learning agents learn policies that complete difficult, procedurally generated platformer levels. The existing Gymnasium environment produces seed-deterministic levels with easy, medium, and hard tiers, a 42-dimensional observation, and eight discrete actions. Previous experiments found that Double DQN led a six-policy benchmark at 39.3% aggregate held-out completion, while a later study found different curriculum preferences for Double DQN and PPO and persistent failures on hard levels. We will compare fixed easy-to-hard training, uniformly mixed difficulty, and an adaptive curriculum for Double DQN and PPO using matched training budgets and multiple independent training seeds. Performance will be measured primarily by completion rate on fresh, paired held-out levels, especially the hard tier. Progress, return, gem collection, stalls, and death causes will help explain outcomes. The goal is to establish whether adaptive training improves robust completion, not to claim that one algorithm is universally best.
Background and Motivation
Reinforcement learning frames this platformer as a Markov decision process: the agent observes a state representation, chooses an action, receives a reward, and transitions to another state. Deep Q-learning combines value-based learning with function approximation and replay, while Double DQN reduces overestimation by separating action selection from target evaluation [1, 2]. Proximal Policy Optimization (PPO) is an on-policy policy-gradient method that limits policy updates to improve training stability [3]. These approaches are already represented in this project, alongside tabular Q-learning, a hand-coded heuristic, random actions, and MCTS. MCTS has access to the simulator during planning and therefore serves as a planning reference rather than a sample-efficiency-matched learner.
The project has built a deterministic, procedurally generated environment with shared increasing-difficulty tiers. This makes paired evaluation possible: policies can be compared on the same level seeds rather than on different layouts. The current benchmark reports Double DQN at 39.3% completion across 150 held-out episodes, ahead of PPO at 24.7%; completion falls for both learned policies at greater difficulty (benchmark summary). The separate five-stage study found that curriculum training helped Double DQN relative to mixed difficulty (27.8% versus 10.7% validation completion), whereas PPO performed better under mixed difficulty (15.9% versus 0%). Its held-out diagnostics found PPO curriculum stalled throughout its 45 evaluated episodes (five-stage summary). Those studies used different experiment designs and evaluation sets, so their rates are informative within each study but should not be combined into one ranking.
These results motivate a focused follow-up: compare a curriculum that can respond to measured learning progress against fixed schedules, while holding the environment, reward setup, and training budget constant. The aim is to address poor hard-tier performance and clarify whether curriculum choice should depend on the learner.
Research question
Does an adaptive difficulty curriculum improve hard-tier completion for Double DQN and PPO compared with fixed easy-to-hard curriculum and uniformly mixed-difficulty training, under matched training budgets?
Hypotheses
H1 (primary): Adaptive curriculum increases hard-tier held-out completion relative to both fixed curriculum and mixed difficulty, averaged across independent training seeds.
H2: Adaptive curriculum does not reduce overall held-out completion by more than five percentage points relative to the better fixed schedule for the same algorithm.
H3 (diagnostic): Any improvement in completion will be accompanied by greater maximum progress and fewer stalled episodes on medium and hard levels. If completion does not improve, progress and death diagnostics will help identify whether the policy is blocked by stalls, falls, hazards, or enemies.
These are hypotheses to test, not conclusions from the existing experiments.
Problem Statement
For each agent and training regimen, learn a policy that maps the environment’s 42-value observation to one of eight movement actions. At training time, the regimen determines which difficulty tier supplies each episode. At test time, evaluate the frozen policy on the same previously unseen level seeds used for the other policies.
The primary outcome is the fraction of hard-tier evaluation episodes completed. A completion occurs when the environment reports that the goal has been reached before the episode terminates or times out. The problem is quantifiable, reproducible, and measurable: level generation and transitions are seed-deterministic, the observation/action spaces are fixed, and each episode yields a completion flag plus progress, return, gem, stall, and death metrics.
Data and Environment
There is no external human-subject or user dataset. The dataset is generated by the project’s simulator through training and evaluation episodes.
Environment: platformer/env.py; Gymnasium-compatible procedural platformer with a normalized 42-value observation and eight discrete actions.
Level tiers: easy, medium, and hard, with progressively wider gaps, narrower platforms, and increased vertical variation and obstacle/moving platform frequency.
Training data: observations, actions, rewards, and transitions generated by interacting with training levels.
Validation data: paired levels used for monitoring and configuration selection; never used for the final reported test estimate.
Test data: fresh level seeds disjoint from all prior training, validation, and test seeds. Before the run, record the exact seed lists in the machine-readable report. Proposed allocation: training seeds 100–199 for model initialization and stochastic training; validation level seeds 1600–1649; final held-out level seeds 2000–2049. Confirm these allocations are unused before starting. Use each held-out seed at all three difficulty tiers so comparisons are paired.
OOD data: a separate set of paired medium seeds using the existing moving_heavy layout variant. It changes moving-platform frequency and motion while preserving standard-seed platform geometry and other obstacles. Proposed held-out layout seeds: 2050–2099, evaluated on both standard and moving-heavy variants at medium difficulty.
The seed ranges above are a proposed extension to the ranges documented for the completed studies. The team must verify that no proposed seed has been used before treating it as fresh test data.
Proposed Solution and Experimental Design
Policies and training regimens
Compare two learned agents already implemented in the repository:
Double DQN, the leading policy in the current multi-policy benchmark.
PPO, included because the previous study showed that its outcome changed substantially with the training regimen.
For each agent, train three regimen variants:
Fixed curriculum: progress through easy, medium, and hard in equal thirds of the environment-step budget.
Uniform mixed: sample easy, medium, and hard uniformly by episode.
Adaptive curriculum: begin on easy and reassess after each block of 50 completed episodes at the current tier. Promote one tier if at least 80% of that block completed; demote one tier if at most 50% completed and a lower tier exists; otherwise stay at the current tier for another block. The schedule changes only the difficulty assigned to future episodes and does not alter reward or transition rules.
The adaptive thresholds and window are fixed before the confirmatory run. Validation may be used for a single pilot to check that the schedule works as implemented, but not to search many thresholds and select the best one. Any tuning beyond that pilot requires a new validation split and must leave the final held-out seeds untouched.
Controls and replication
Use the same environment options, reward shaping, obstacle settings, episode horizon, and training-step budget for all six conditions.
Train each condition for 25,000 environment steps per run, matching the completed benchmark’s neural training budget.
Use five independent training seeds per condition (30 trained checkpoints total). Pair seed numbers across conditions where practical; log the actual seeds, steps, and tier exposure.
Evaluate each checkpoint on 50 paired level seeds per tier (150 episodes per checkpoint). This gives 750 episodes per condition and 4,500 episodes across all six conditions, before the separate OOD evaluation.
Evaluate every frozen checkpoint on identical held-out seeds. Do not select best-attempt episodes as a substitute for aggregate metrics.
Include random and heuristic policies as sanity-check references. Report MCTS separately because it queries the simulator during planning.
Preserve the current benchmark and five-stage outputs as historical results; the new experiment gets its own directory and report.
Analysis
The primary comparison is hard-tier completion by training regimen within each policy. Report each training seed’s result, the mean, and a 95% confidence interval. Because episodes are clustered within trained checkpoints and level seeds are paired across conditions, use a hierarchical bootstrap that resamples training seeds and then paired level seeds. Also report a paired regimen difference with its interval. Do not treat thousands of episodes as thousands of independent training runs.
Analyze overall and tier-specific completion, mean episodic return, mean maximum progress, gem fraction, stall fraction, deaths by cause, and time-limit truncations. Report OOD standard-to-moving-heavy completion gaps separately. Include training time and environment steps so compute is transparent. Keep all seed lists, configuration values, checkpoints, and per-episode results in versioned machine-readable artifacts.
Evaluation Metrics
For a set of (N) evaluation episodes, let (C_i=1) if episode (i) completes and (0) otherwise. Completion rate is:
[  = _{i=1}^{N} C_i. ]
The primary metric is completion rate on hard-tier held-out levels. The secondary overall rate is computed over equal numbers of easy, medium, and hard episodes, so easy levels cannot dominate through a larger sample.
For each episode, maximum progress is the furthest horizontal progress normalized by the goal distance; report its mean by difficulty. Gem fraction is gems collected divided by available gems. Stall fraction is the fraction of steps classified as stalls by the existing environment diagnostic. Episodic return is the sum of rewards in an episode. Death counts and time-limit truncations are reported as counts and as fractions of episodes.
For OOD evaluation:
[  = {} - {}. ]
For each agent and regimen, report the mean across training seeds and a 95% interval that respects both training-seed variation and paired level seeds. A confidence interval crossing zero means the observed difference is uncertain at the chosen interval level; it is not proof that the methods are identical.
Expected Results and Success Criteria
The project is successful if it produces a reproducible, statistically transparent comparison, including if the adaptive curriculum does not win. Evidence supporting H1 would be a higher adaptive hard-tier completion mean than both fixed schedules with a paired interval for the improvement above zero. H2 is supported if the lower bound for adaptive-minus-best-fixed overall completion is above -0.05. If intervals include zero or the non-inferiority threshold, report the comparison as inconclusive rather than declaring a winner.
Based on prior results, a plausible outcome is that Double DQN benefits from progression while PPO needs more balanced exposure or an adaptive schedule. This is a motivation for the experiment, not an expected result that should influence reporting.
Ethics, Privacy, and Limitations
The project uses a synthetic game environment and does not collect personal, behavioral, or sensitive data from people. The main risks are methodological: overfitting to a procedural generator, selecting favorable seeds, conflating training-seed variation with episode-level uncertainty, and presenting simulator-aware MCTS as directly comparable to model-free agents. Mitigate these by using paired seeds, fresh held-out evaluation, explicit seed accounting, uncertainty intervals at the correct unit of replication, and a separate MCTS interpretation.
The environment is a simplified platformer, so results do not establish real-world robot or human safety. A 25,000-step training budget and five training seeds per condition remain modest. Conclusions will be scoped to this simulator and these configurations.
Team Expectations
Assign one owner for experiment code and environment changes, one for reproducibility and data/seed tracking, and one for analysis and reporting; team members review one another’s work.
Keep experiment settings, decisions, and blockers in the repository or shared project notes. Do not change the confirmatory protocol without recording why and which results are affected.
Run code review and tests before merging changes that affect environment dynamics, evaluation, or metrics.
Share results and unexpected failures promptly. Resolve disagreements by consulting the pre-registered metric and protocol, then document any decision.
Protect focused work time and agree on milestone owners and delivery dates at the start of the project.
Proposed Timeline
The schedule is relative to project start so the team can align it with the course calendar.
Week
Work before check-in
Check-in decision / deliverable
1
Review the environment, existing benchmark, five-stage report, and proposal; confirm team roles and seed allocations.
Lock research question, hypotheses, and the fresh seed inventory.
2
Implement the adaptive schedule and add tests for tier transitions, episode accounting, and deterministic evaluation.
Review a small smoke run; freeze schedule thresholds and budgets.
3
Run a pilot on validation seeds; verify paired evaluation and reporting.
Approve the confirmatory protocol; no changes to final test seeds.
4
Train all six conditions with five seeds and matched budgets; save checkpoints and logs.
Audit run completeness, tier exposure, failures, and compute.
5
Run held-out and moving-heavy tests; calculate hierarchical intervals and diagnostics.
Review the results and classify findings as supported, uncertain, or unsupported.
6
Produce charts, replay examples, report, and reproducibility instructions; rerun tests.
Final presentation and delivery of source, data, and limitations.

Repository Materials and References
Project evidence
README.md: environment, algorithm families, seed protocol, and study instructions.
experiments/SUMMARY.md and experiments/results.json: six-policy increasing-difficulty benchmark, selected configurations, and held-out results.
experiments/iterative_study/SUMMARY.md and experiments/iterative_study/results.json: curriculum, seed-variation, layout-generalization, ablation, and diagnostic results.
experiments/visual_report.png: rendered overview of benchmark and follow-up findings.
Background literature
Mnih, V. et al. (2015). “Human-level control through deep reinforcement learning.” Nature, 518, 529–533. https://doi.org/10.1038/nature14236
van Hasselt, H., Guez, A., and Silver, D. (2016). “Deep Reinforcement Learning with Double Q-learning.” Proceedings of the AAAI Conference on Artificial Intelligence, 30(1). https://doi.org/10.1609/aaai.v30i1.10295
Schulman, J. et al. (2017). “Proximal Policy Optimization Algorithms.” arXiv:1707.06347. https://arxiv.org/abs/1707.06347
Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd edition. MIT Press. http://incompleteideas.net/book/the-book-2nd.html


