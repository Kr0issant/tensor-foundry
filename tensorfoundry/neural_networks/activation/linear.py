import numpy as np
from .base import Base

class Linear(Base):
    def compute(x: float | np.ndarray):
        return x

    def derivative(x: float | np.ndarray):
        return np.ones_like(x)
