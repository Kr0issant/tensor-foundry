import numpy as np

EPSILON = 1e-15

class Base:
    def compute(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        raise NotImplementedError("Subclasses must implement compute()")

    def derivative(y_true: float | np.ndarray, y_pred: float | np.ndarray):
        raise NotImplementedError("Subclasses must implement derivative()")
