"""Lightweight installation diagnostics that do not create a simulator."""

import argparse
import importlib
import platform
import shutil
import sys


REQUIRED_MODULES = (
    "isaacgym",
    "numpy",
    "torch",
    "gym",
    "hydra",
    "omegaconf",
)

REAL_ROBOT_MODULES = (
    "cv2",
    "open3d",
    "pyaudio",
    "pyrealsense2",
    "zmq",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Also validate optional camera, audio, RealSense, and ZMQ imports.",
    )
    args = parser.parse_args(argv)

    failures = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    module_names = REQUIRED_MODULES + (REAL_ROBOT_MODULES if args.real_robot else ())
    imported_modules = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            failures.append((module_name, str(error)))
            print(f"[FAIL] {module_name}: {error}")
            continue
        imported_modules[module_name] = module
        version = getattr(module, "__version__", "available")
        print(f"[ OK ] {module_name}: {version}")

    if args.real_robot and "cv2" in imported_modules:
        if not hasattr(imported_modules["cv2"], "aruco"):
            failures.append(("cv2.aruco", "install opencv-contrib-python"))
            print("[FAIL] cv2.aruco: install opencv-contrib-python")
        else:
            print("[ OK ] cv2.aruco: available")

    if args.real_robot:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            print(f"[ OK ] ffmpeg: {ffmpeg}")
        else:
            print("[WARN] ffmpeg: unavailable; compressed demo recording is disabled")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except Exception:
        pass

    if failures:
        print("\nInstallation check failed.")
        return 1
    scope = "Core and real-robot" if args.real_robot else "Core"
    print(f"\n{scope} Python dependencies are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
