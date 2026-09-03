"""Timing helpers for control loops and human-readable status output."""

import time


def remaining_sleep(start_time, duration, verbose=True):
    """Sleep for the unspent part of a fixed-duration control interval."""
    remaining_time = duration - (time.perf_counter() - start_time)
    if remaining_time > 0:
        time.sleep(remaining_time)
    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"Control Frequency: {1 / elapsed:.3f}Hz")


def convert_time(relative_time):
    """Format a duration in integer seconds as hours:minutes:seconds."""
    relative_time = int(relative_time)
    hours = relative_time // 3600
    left_time = relative_time % 3600
    minutes = left_time // 60
    seconds = left_time % 60
    return f"{hours}:{minutes}:{seconds}"


def format_timestamp(value):
    """Format signed seconds as minutes:seconds.milliseconds."""
    negative = value < 0
    value = abs(value)
    minutes, seconds = divmod(int(value), 60)
    milliseconds = int((value - int(value)) * 1_000)
    prefix = "-" if negative else ""
    return f"{prefix}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
