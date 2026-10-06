import numpy as np

class Base:
    def step(weights: list[np.ndarray], biases: list[np.ndarray], d_weights: list[np.ndarray], d_biases: list[np.ndarray], learning_rate: float):
        raise NotImplementedError("Subclasses must implement step()")

    def batch(input: list[np.ndarray], target: list[np.ndarray], num_samples: int):
        raise NotImplementedError("Subclasses must implement batch()")
