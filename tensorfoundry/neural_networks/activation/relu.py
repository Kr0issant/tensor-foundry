import numpy as np
from .base import Base

class ReLU(Base):
    def compute(x: float | np.ndarray):
        return np.maximum(0, x)

    def derivative(x: float | np.ndarray):
        return (x > 0).astype(float)
