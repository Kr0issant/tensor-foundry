import numpy as np
from .base import Base

class MiniBatchSGD(Base):
    def step(weights: list[np.ndarray], biases: list[np.ndarray], d_weights: list[np.ndarray], d_biases: list[np.ndarray], learning_rate: float):
        for i in range(len(weights)):
            weights[i] -= learning_rate * d_weights[i]
            biases[i] -= learning_rate * d_biases[i]
        return weights, biases

    def batch(input: list[np.ndarray], target: list[np.ndarray], num_samples: int, batch_size: int):
        permutation = np.random.permutation(num_samples)
        input_shuffled = input[permutation]
        target_shuffled = target[permutation]
            
        return [(input_shuffled[i:i+batch_size], target_shuffled[i:i+batch_size]) for i in range(0, num_samples, batch_size)]
