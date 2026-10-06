import numpy as np
from .base import Base

class Softmax(Base):
    def compute(x: float | np.ndarray):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(x: float | np.ndarray):
        s = Softmax.compute(x)
        return s * (1 - s)
