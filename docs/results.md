# Paper evaluation and figures

The release keeps machine-specific and dated artifact identifiers out of
executable code. Three versioned configurations under
`projects/TimeAwarePolicy/paper/configs/` describe the available paper
workflows:

| Definition | Scope |
|---|---|
| `policy_evaluation.json` | Strict time-optimal and CMDP evaluation jobs |
| `stagewise_stability_evaluation.json` | Paired staged/constant stability jobs and schedules |
| `figure_generation.json` | Available curve inputs, tables, stability status, and CMDP plots |

Every generated manifest records the profile path and SHA-256 hash. Changing
an artifact, checkpoint, schedule, seed, or evaluation count therefore
requires a visible profile change.

## 1. Verify inputs

The small demo bundle included in Git can be checked directly:

~~~bash
bash tests/release/check.sh
~~~

The paper-workflow definitions refer to the separate archived manuscript-result
bundle, which contains more checkpoints and campaign histories than the six
compact demo policies. Point the launchers at that bundle and run preflight
before allocating GPUs:

~~~bash
python -m projects.TimeAwarePolicy.evaluation.launch_policy_evaluations \
  --profile projects/TimeAwarePolicy/paper/configs/policy_evaluation.json \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res \
  --output-dir /path/to/evaluation_output \
  --preflight-only

python -m projects.TimeAwarePolicy.evaluation.launch_stagewise_stability \
  --profile projects/TimeAwarePolicy/paper/configs/stagewise_stability_evaluation.json \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res \
  --output-dir /path/to/stagewise_stability_campaign \
  --preflight-only full
~~~

Preflight fails on missing or ambiguous artifact directories. It does not
launch a simulator or modify the source artifact stores.

## 2. Run and validate stagewise stability

After a successful preflight, launch the short canary first and then the full
paired campaign:

~~~bash
python -m projects.TimeAwarePolicy.evaluation.launch_stagewise_stability \
  --launch canary --gpus 0,1,2 \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res \
  --output-dir /path/to/stagewise_stability_campaign

python -m projects.TimeAwarePolicy.evaluation.launch_stagewise_stability \
  --launch full --gpus 0,1,2 \
  --train-res-dir /path/to/train_res \
  --eval-res-dir /path/to/eval_res \
  --output-dir /path/to/stagewise_stability_campaign

python -m projects.TimeAwarePolicy.evaluation.validate_stagewise_stability \
  /path/to/stagewise_stability_campaign/full_fullstable_repeat2_status.json
~~~

The full protocol uses the fixed 1,000-configuration bank twice per
controller. It calculates both the full-stable-stage time-average and peak
instantaneous object-motion proxies. The distance plot uses 20 equal-width Cube
manipulation-distance bins with a mean curve and one-standard-deviation shaded
region. Plotting uses both-success paired rollouts only.

## 3. Build available figures and tables

`paper/build_results.py` separates campaign selection (`datasets.py`), curve
aggregation (`curves.py`), stagewise-stability logic
(`figures/stagewise_stability.py`), and reporting (`reports.py`). To build the
full available package:

~~~bash
python -m projects.TimeAwarePolicy.paper.build_results \
  --profile projects/TimeAwarePolicy/paper/configs/figure_generation.json \
  --output-dir /path/to/final_three_task \
  --review-cube-dir /path/to/cube_quality_campaign \
  --include-stagewise-stability \
  --stagewise-stability-status /path/to/stagewise_stability_campaign/full_fullstable_repeat2_status.json
~~~

The output directory must already contain the four curve-input CSVs and the
strict time-optimal table at the relative locations declared in
`figure_generation.json`. Available outputs cover the time-optimal curves, CMDP
curves and tables, the real-single-seed CMDP view, and both stability metric
candidates. The repository does not claim to reconstruct figures whose raw
inputs are unavailable.

For the stagewise-stability analysis alone:

~~~bash
python -m projects.TimeAwarePolicy.plot stagewise-stability \
  --output-dir /path/to/final_three_task \
  --stagewise-stability-status /path/to/stagewise_stability_campaign/full_fullstable_repeat2_status.json
~~~

The builder emits selected-history CSVs, aggregate CSVs, source manifests,
profile provenance, and validation JSON alongside the PDFs/PNGs. No manuscript
source files are changed.
