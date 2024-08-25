import squigglepy as sq
import numpy as np

def sample_or_broadcast(value, m):
    if isinstance(value, sq.LognormalDistribution):
        return sq.sample(value, n=m)
    elif isinstance(value, (int, float)):
        return np.full(m, value)
    elif isinstance(value, np.ndarray) and value.shape[0] == m:
        return value
    else:
        raise ValueError(f"Unexpected type for value: {type(value)}")
