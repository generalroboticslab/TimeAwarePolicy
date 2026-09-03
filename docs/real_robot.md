# Real-robot deployment

The `real_robot/` package is a maintained part of this release. It contains
the RealSense interface, ArUco-based task state estimators, optional video and
audio recording, the hardware evaluation lifecycle, and the ZMQ client used by
the Franka controller. Hardware dependencies remain lazy and these files are
not required for simulation-only training or evaluation.

## Hardware and software prerequisites

- A Franka system with the project controller service already installed and
  tested independently.
- An Intel RealSense camera and matching system drivers.
- Printed ArUco markers whose physical dimensions match the selected task's
  estimator constants.
- A calibrated 4-by-4 base-to-camera transform.
- PortAudio development/runtime libraries for optional audio recording and
  FFmpeg for compressed demo video.

Install and check the optional Python dependencies without opening a camera or
connecting to the controller:

~~~bash
pip install -r requirements-real.txt --no-cache-dir
bash tests/release/check.sh --real-robot
~~~

The dependency check verifies imports and the OpenCV ArUco module. It cannot
verify camera permissions, calibration, network reachability, robot limits, or
the external Franka controller stack.

## Controller protocol

`real_robot.SocketClient.FrankaClient` uses two ZMQ sockets:

- a conflated subscriber on `--controller_sub_port` (default 5555) for the
  most recent robot state;
- a publisher on `--controller_pub_port` (default 5556) for control commands.

The controller host must be supplied explicitly with `--controller_ip`; no
institution-specific address is embedded in the public code. The external
controller must use the same Python-object message schema expected by
`FrankaClient` and the task environment.

Python-object ZMQ messages use pickle serialization. Deserializing messages
from an untrusted sender can execute arbitrary code, so this protocol must only
be used with the intended controller on a trusted, isolated network. Do not
expose either controller port to the public internet or an untrusted LAN. A
future protocol change must update both this client and the external controller
to use authenticated, schema-validated messages.

## Calibration and observation-only check

`--cam_ext_path` defaults to `cal_results/franka2cam.txt`. The file must contain
the 16 whitespace-separated entries of a 4-by-4 homogeneous transform, one
matrix row per line. Confirm marker sizes, marker IDs, transform direction, and
the task-specific residual offsets in `real_robot/StateEstimator.py` for your
physical setup.

Before enabling commands, perform an observation-only run:

~~~bash
bash exec/TimeAwarePolicy/eval/run.sh \
  --num_envs 1 --real_robot --not_move \
  --controller_ip CONTROLLER_HOST \
  --cam_ext_path /absolute/path/to/franka2cam.txt \
  --checkpoint CHECKPOINT --index_episode best_rew \
  --par_configs_eval true \
  --goal_time 10
~~~

`--not_move` suppresses command publication, but it still starts the camera,
connects to the state stream, constructs the simulator, and executes policy
inference. `--par_configs_eval true` loads the time-aware checkpoint's matching
temporal calibration bank. Inspect estimated poses and generated commands
before removing `--not_move`.

## Enabling motion and recording

After the observation-only check, remove `--not_move` to publish commands. To
record a demonstration, add a unique `--demo_name`; recordings are written
under `--demo_dir` (default `demos`). Select the OpenCV recording device with
`--demo_camera_index` (default 0). The RealSense camera used for state
estimation is separate from this optional recorder.

## Safety checklist

Real-robot policies can cause injury or equipment damage. Before every motion
run:

1. verify the emergency stop and external controller watchdog;
2. clear the workspace and enforce robot-side joint, Cartesian, velocity, and
   torque limits;
3. validate the camera transform and object pose estimates in `--not_move`
   mode;
4. confirm that the selected checkpoint, task, control type, and physical
   objects match;
5. begin at conservative robot-side gains and maintain an operator stop path.

The Python client is not a safety controller. Network loss, stale perception,
or an exception must be handled safely by the external Franka controller.
