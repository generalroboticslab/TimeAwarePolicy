# Included checkpoints

The repository includes one initializer and one final time-aware policy for
each task, plus the matching 1,000-configuration temporal calibration bank.

| Task | Vanilla checkpoint | Time-aware checkpoint |
|---|---|---|
| Cube stacking | 20250716_142057_va_FrankaCubeStack | 20250717_162724_tw_FrankaCubeStack |
| GM pouring | 20250703_164406_va_FrankaGmPour | 20250715_123940_tw_FrankaGmPour |
| Drawer opening | 20250730_012004_va_FrankaCabinet | 20250730_151924_tw_FrankaCabinet |

The initializer folders contain init, last, best-success, and best-score policy
snapshots plus matching reward normalizers. The time-aware folders contain
last, best-success, and best-score snapshots. Use index_episode best_rew for
the public demos.

The machine-readable source of truth is
`projects/TimeAwarePolicy/paper/configs/bundled_files.json`. It gives the exact directory,
selected checkpoint, calibration bank, SHA-256 checksum, and expected bank size
for every task. Verify all 24 required files with:

~~~bash
bash tests/release/check.sh
~~~

These are historical paper artifacts. Their config.json files are the source
of truth for how each checkpoint was produced, including critic reset,
punctuality scale, and student clock behavior. The evaluator restores their
recorded architecture/controller settings before applying explicit evaluation
overrides.

The bundled initializer folders are the maintained public starting-policy
artifacts. Historical time-aware `config.json` files may name an earlier
temporal-student/evaluation producer used in the original run; those archived
producer checkpoints are not silently reconstructed or relabeled. The fixed
calibration banks required by the public time-aware demos are included and
verified separately by the artifact manifest.

Do not overwrite included folders. New runs receive timestamped names and
should be stored in a separate result root when comparing implementations.
