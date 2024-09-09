import numpy as np


def ensure_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise TypeError(f'invalid type `{type(x)}`')
