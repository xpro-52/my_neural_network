import numpy as np


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum((y - t) ** 2)
