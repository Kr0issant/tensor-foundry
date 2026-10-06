import numpy as np
from .base import Base

EPSILON = 1e-15

class CategoricalCrossEntropy(Base):
    def compute(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    def derivative(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -(y_true / y_pred)
    