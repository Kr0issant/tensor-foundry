import numpy as np
from .base import Base

class SGD(Base):
    def step(weights: list[np.ndarray], biases: list[np.ndarray], d_weights: list[np.ndarray], d_biases: list[np.ndarray], learning_rate: float):
        for i in range(len(weights)):
            weights[i] -= learning_rate * d_weights[i]
            biases[i] -= learning_rate * d_biases[i]
        return weights, biases

    def batch(input: list[np.ndarray], target: list[np.ndarray], num_samples: int):
        permutation = np.random.permutation(num_samples)
        input_shuffled = input[permutation]
        target_shuffled = target[permutation]
            
        return [(input_shuffled[i:i+1], target_shuffled[i:i+1]) for i in range(num_samples)]
