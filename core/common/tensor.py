"""Framework-light conversion helpers."""

import numpy as np


def to_numpy(value):
    """Convert a tensor or array-like value to a NumPy array."""
    if all(hasattr(value, attribute) for attribute in ("detach", "cpu", "numpy")):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return np.expand_dims(value, axis=0) if value.ndim == 0 else value
