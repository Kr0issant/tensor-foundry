import numpy as np

class Base:
    def compute(x: float | np.ndarray):
        raise NotImplementedError("Subclasses must implement compute()")

    def derivative(x: float | np.ndarray):
        raise NotImplementedError("Subclasses must implement derivative()")
