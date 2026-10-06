import numpy as np
from enum import Enum

class WeightInitializer(Enum):
    ZERO = 0
    RANDOM = 1
    XAVIER = 2
    HE = 3

    def get_weights(self, layers: np.ndarray):
        match self:
            case WeightInitializer.ZERO:
                return [np.zeros(size=(l[0], l[1])).astype(np.float32) for l in layers]
            case WeightInitializer.RANDOM:
                return [np.random.normal(loc=0.0, scale=1.0, size=(l[0], l[1])).astype(np.float32) for l in layers]
            case WeightInitializer.XAVIER:
                return [np.random.normal(loc=0.0, scale=np.sqrt(2.0 / (l[0] + l[1])), size=(l[0], l[1])).astype(np.float32) for l in layers]
            case WeightInitializer.HE:
                return [np.random.normal(loc=0.0, scale=np.sqrt(2.0 / l[0]), size=(l[0], l[1])).astype(np.float32) for l in layers]
            case _:
                raise ValueError(f"Unsupported weight initialization method: {self}")
