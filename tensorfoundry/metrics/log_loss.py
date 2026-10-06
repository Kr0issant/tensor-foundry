import numpy as np
from .base import Base

EPSILON = 1e-15

class LogLoss(Base):
    def compute(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    