# Reproducibility

## Determinism

Training seeds Python, NumPy, and PyTorch and enables deterministic cuDNN
behavior by default. Isaac Gym GPU physics can still exhibit platform- and
driver-dependent numerical variation. Record the GPU model, driver, CUDA
runtime, command, seed, code commit, and checkpoint hashes for every reported
campaign.

## Domain randomization

Randomization expands through a curriculum. Ratio 0 uses the narrowest
settings; ratio 1 uses the configured full range. Full observation noise
includes:

- source position ±0.01 m and source orientation ±π/60 rad;
- source-to-target relative position ±0.01 m for Cube and GM Pour;
- drawer handle position ±0.01 m;
- end-effector position ±0.01 m and orientation ±π/60 rad;
- current arm joint position ±π/60 rad.

Target poses are spatially randomized, but target orientation is not a
separate actor observation for Cube or GM Pour. Controller randomization uses
gripper velocity noise ±0.005 m/s and a sampled 0.1–0.3 s delay after a
gripper-mode change is detected.

## Task semantics

The public default uses the manuscript's 5 cm end-effector clearance in Cube
success. Cabinet has no general contact-force failure condition because
intentional handle grasping produces large contact forces. The Cube clearance
is serialized in new run configurations.

## Calibration-bank sampling

The 1,000-entry fixed calibration banks supply configuration-specific minimum
completion times and instability thresholds. Simulation evaluation currently
samples these bank indices with replacement. If a study needs distinct
physical configurations, generate and calibrate a larger bank or implement a
separately versioned evaluation protocol rather than silently changing the
denominator.

The versioned stagewise-stability protocol is that separately defined exception. It
enumerates all 1,000 entries exactly twice per controller and validates exact
pairing before plotting. Its two controller schedules use the same
configuration-specific `T_min`, with `T_goal = 2*T_min`.

## Artifact integrity

Run `bash tests/release/check.sh` before checkpoint evaluation.
The release tests load all six bundled policies strictly against their recorded
MLP layouts. The verifier also checks each selected policy, reward normalizer,
saved config, and calibration-bank files against
`projects/TimeAwarePolicy/paper/configs/bundled_files.json`, then
checks that every bank field contains exactly 1,000 entries.
