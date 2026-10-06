import numpy as np
from .base import Base

class Sigmoid(Base):
    def compute(x: float | np.ndarray):
        return 1 / (1 + np.exp(-x))

    def derivative(x: float | np.ndarray):
        s = Sigmoid.compute(x)
        return s * (1 - s)
