# Evaluation

## Strict 2,000-rollout evaluation

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh \
  --saving --graphics_device_id -1 \
  --num_envs 2000 --target_success_eps 2000 --strict_eval \
  --checkpoint CHECKPOINT --index_episode best_rew
~~~

Strict mode gives every environment one rollout and then removes it from the
active set. The reported denominator is therefore exactly 2,000.

For a time-aware policy evaluated over the saved temporal calibration bank:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh \
  --saving --graphics_device_id -1 \
  --num_envs 2000 --target_success_eps 2000 --strict_eval \
  --par_configs_eval true \
  --goal_ratio_range "[0.2, 1.0, 0.1]" \
  --checkpoint CHECKPOINT --index_episode best_rew
~~~

The current fixed-bank implementation samples reference indices uniformly
with replacement. A 2,000-rollout evaluation over a 1,000-entry bank therefore
does not contain 2,000 distinct bank entries. Record this fact in result
provenance.

This generic behavior does not apply to the paired stagewise-stability
protocol. It explicitly evaluates bank indices 0–999 twice, in identical
order for constant-ratio and staged controllers. Its output is therefore
2,000 runs per controller over 1,000 distinct fixed configurations.

## Metrics

Time-optimal policies:

- success rate over every rollout;
- completion time averaged only over successful episodes.

Time-aware CMDP policies:

- success rate over every rollout;
- absolute punctuality mismatch averaged only over successful episodes;
- cumulative and peak scene-instability metrics averaged only over successful
  episodes.

Failed episodes are never represented as zero-time successes. Training and
evaluation both persist eps_time_p as a success-conditioned metric.

## Robustness conditions

Cube restitution:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh ... --add_restitution
~~~

Cube disturbance:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh ... --apply_disturbances --disturbance_v 10
~~~

Forty-particle pouring:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh ... --num_gms_eval 40
~~~

Cabinet friction or payload:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh ... --friction_mul 2
bash exec/TimeAwarePolicy/eval/run.sh ... --num_props_eval 6
~~~

## Curves

~~~bash
bash exec/TimeAwarePolicy/plot/run.sh training-curves \
  --campaign-dir /path/to/campaign
~~~

The campaign directory must contain status.json, its recorded per-run logs,
and (for timing series) its offline W&B stream. The plotting utility uses
attempted update for every series, normalizes the stored punctuality fields,
and keeps success-conditioned time metrics separate from all-episode success.

## Available paper-result utilities

The public result utilities live under `projects/TimeAwarePolicy/`:

- `evaluation/launch_policy_evaluations.py` orchestrates strict 2,000-rollout policy
  evaluations and records a provenance manifest;
- `evaluation/launch_stagewise_stability.py` orchestrates paired stable-stage
  evaluation on a fixed 1,000-configuration bank, repeated twice per configuration;
- `evaluation/validate_stagewise_stability.py` checks those paired artifacts;
- `paper/build_results.py` builds the available evaluation figures and
  summary tables; and
- `paper/figures/cmdp_solver_comparison.py` builds the available multi-seed
  CMDP curves.

Inspect an entrypoint with `python -m MODULE --help`. The launchers require the
corresponding checkpoint and calibration-bank artifact bundle; the six compact
demo checkpoints included in this repository are documented separately in
`docs/checkpoints.md` and are not silently substituted for missing experiment
artifacts. Output roots are explicit command-line options rather than
machine-specific paths.

Before launching GPU work, resolve every declared artifact without changing
external state:

~~~bash
python -m projects.TimeAwarePolicy.evaluation.launch_policy_evaluations \
  --preflight-only \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res

python -m projects.TimeAwarePolicy.evaluation.launch_stagewise_stability \
  --preflight-only full \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res
~~~

Both preflight commands print validated JSON containing the profile path/hash
and every resolved input. Actual launches persist the corresponding provenance
manifests. See `docs/results.md` for the full reconstruction workflow.
