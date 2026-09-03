"""Lightweight installation diagnostics that do not create a simulator."""

import argparse
import importlib
from importlib import metadata
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

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def _installed_opencv_distributions():
    installed = {}
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
    return installed


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

    if args.real_robot:
        opencv_distributions = _installed_opencv_distributions()
        if list(opencv_distributions) != ["opencv-python"]:
            detail = ", ".join(
                f"{name}=={version}"
                for name, version in opencv_distributions.items()
            ) or "none"
            failures.append(("OpenCV distribution", detail))
            print(
                "[FAIL] OpenCV distribution: expected only opencv-python; "
                f"found {detail}"
            )
        else:
            version = opencv_distributions["opencv-python"]
            print(f"[ OK ] OpenCV distribution: opencv-python=={version}")

    if args.real_robot and "cv2" in imported_modules:
        aruco = getattr(imported_modules["cv2"], "aruco", None)
        if aruco is None or not hasattr(aruco, "ArucoDetector"):
            failures.append(("cv2.aruco.ArucoDetector", "unavailable"))
            print("[FAIL] cv2.aruco.ArucoDetector: unavailable")
        else:
            print("[ OK ] cv2.aruco.ArucoDetector: available")

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
