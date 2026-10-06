import numpy as np
from .base import Base

class MeanSquaredError(Base):
    def compute(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        return (y_true - y_pred) ** 2

    def derivative(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        return 2 * (y_pred - y_true)
