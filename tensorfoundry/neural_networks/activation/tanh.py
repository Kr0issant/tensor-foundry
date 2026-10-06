import numpy as np
from .base import Base

class Tanh(Base):
    def compute(x: float | np.ndarray):
        return np.tanh(x)

    def derivative(x: float | np.ndarray):
        return 1 - (np.tanh(x) ** 2)
