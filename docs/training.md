# Training protocol

Run commands from the repository root. TASK is one of FrankaCubeStack,
FrankaGmPour, or FrankaCabinet.

Training enables W&B logging when results are saved. No account or team is
hard-coded: use `--wandb_entity TEAM` and `--wandb_project PROJECT` when
needed, or set `WANDB_MODE=offline` for a self-contained local run.

The commands below are an explicit, current-code recipe for new runs. They do
not rewrite the provenance of included historical checkpoints; each saved
`config.json` is authoritative for the run that produced it.

## Training budgets

| Stage | Cube stacking | Granular pouring | Drawer opening |
|---|---:|---:|---:|
| Initial task policy | 2,500 | 6,000 | 2,500 |
| Time-optimal fine-tuning | 1,500 | 2,500 | 1,500 |
| Temporal-input distillation | 1,500 | 2,500 | 1,500 |
| Time-aware CMDP fine-tuning | 1,500 | 2,500 | 1,500 |

The initial-policy budgets are task-specific caps based on the saved producer
histories. The four initializer-quality checkpoints are selected from this
same initial-policy run; they do not require a separate training stage. GM Pour
uses the larger downstream budget because it converges more slowly. Retain a
distilled student only after its strict success remains within 5 percentage
points of its time-optimal teacher.

## Task settings

| Task | Initial/time-optimal/student horizon | Time-aware horizon |
|---|---:|---:|
| FrankaCubeStack | 500 | 2000 |
| FrankaGmPour | 500 | 1600 |
| FrankaCabinet | 800 | 2600 |

## 1. Initial task policy

~~~bash
bash exec/TimeAwarePolicy/train/initial_policy.sh \
  --task "$TASK"
~~~

Initial-policy training starts the domain-randomization curriculum at ratio 0.
After ten qualifying records above 90% rolling success, it expands the ratio
by 0.03. Once rolling success first reaches 90% while the ratio is still 0,
the command above saves an **unlabeled** candidate immediately and every five
accepted updates thereafter. Rejected updates do not need another snapshot
because they do not change the policy.

After training, preview the held-out evaluation commands and then execute them:

~~~bash
bash exec/TimeAwarePolicy/train/initializer_quality.sh \
  --task "$TASK" \
  --producer "$INITIAL_TRAINING_FOLDER" \
  --output-bank "$INITIALIZER_SET"

bash exec/TimeAwarePolicy/train/initializer_quality.sh \
  --task "$TASK" \
  --producer "$INITIAL_TRAINING_FOLDER" \
  --output-bank "$INITIALIZER_SET" \
  --execute
~~~

The first command is a dry-run: it prints the strict-evaluation commands
without launching them. `--execute` runs the evaluations and creates
`INITIALIZER_SET`, a folder containing the selected Q40/Q60/Q80/Q95
checkpoints and their measured-success metadata. This initializer checkpoint
set is separate from the temporal calibration bank created after student
distillation.

Each candidate is measured on the same 2,000 newly sampled configurations at
full domain randomization, with observation/action noise enabled and no fixed
reference bank. Evaluation proceeds in update order until measured success
reaches 95%, or until all candidates are exhausted. Four distinct candidates
are assigned to the nominal 40/60/80/95 targets by minimum total absolute
error. The resulting metadata reports the actual strict success of every
candidate and selection; a q-label is never treated as measured performance.

## 2. Time-optimal fine-tuning

~~~bash
bash exec/TimeAwarePolicy/train/time_optimal.sh \
  --task "$TASK" \
  --checkpoint "$INITIAL_CHECKPOINT" \
  --index best
~~~

## 3. Temporal-input distillation

~~~bash
bash exec/TimeAwarePolicy/train/temporal_student.sh \
  --task "$TASK" \
  --checkpoint "$TIME_OPTIMAL_CHECKPOINT"
~~~

For student training, time_ratio defaults to true. The sampled time ratio
therefore controls the remaining-time clock.

## Temporal calibration bank

~~~bash
bash exec/TimeAwarePolicy/eval/temporal_bank.sh \
  --checkpoint "$TEMPORAL_STUDENT"
~~~

The command follows the student's saved parent pointer and uses the
time-optimal policy—not the student policy—to collect 1,000
configuration–reference pairs containing minimum-time and instability
statistics. Passing the student checkpoint keeps this calibration bank
associated with the matching student so the subsequent time-aware stage can
load it directly. Results are stored under eval_res.

## 4. Time-aware CMDP fine-tuning

~~~bash
bash exec/TimeAwarePolicy/train/time_aware.sh \
  --task "$TASK" \
  --checkpoint "$TEMPORAL_STUDENT"
~~~

Set cmdp_method to np3o, ppo_lagrangian, or cpo. The reward critic and all
cost critics are newly initialized; all remain critic-only for the first 50
attempted updates. PPO-Lagrangian also freezes its multiplier during warmup.
CPO uses its own max-KL trust region (default 0.01).

The selected result artifacts were produced across multiple experiment
generations, so not every archived checkpoint used this exact reset/warmup
recipe. Use its saved `config.json` when reproducing or describing that
specific artifact.

## Common optimization settings

| Setting | Value |
|---|---:|
| PPO learning rate | 0.0002 |
| Parallel environments | 16,384 |
| Control frequency | 20 Hz |
| Rollout steps per update | 32 |
| Rollout batch size | 524,288 |
| Minibatch size | 131,072 |
| Update epochs | 5 |
| Reward discount | 0.995 |
| GAE lambda | 0.95 |
| PPO clip coefficient | 0.2 |
| Reward/cost critic coefficient | 0.5 |
| Entropy coefficient | 0.005 |
| Gradient-norm limit | 0.5 |
| PPO target KL | 2.5 |
| Hidden widths | [256, 128, 64] |
| Task-success reward scale | 1,000 |

These are the public CLI defaults and match the manuscript's default PPO
table. Stage commands above state their necessary overrides, including the
distillation learning rate and time-aware reward discount.

The readable shell entrypoints above contain the complete public recipe and
forward any additional arguments. Their shared implementation is
[`projects/TimeAwarePolicy/train.py`](../projects/TimeAwarePolicy/train.py); an
explicit `--num_updates` or `--total_timesteps` takes precedence. Every
training log records both attempted and accepted updates; plots use attempted
updates on the x-axis.
