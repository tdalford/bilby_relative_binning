import numpy as np


def null_jacobian(sample):
    """Dummy jacobian function, returns 1"""
    return np.ones_like(list(sample.values())[0])
