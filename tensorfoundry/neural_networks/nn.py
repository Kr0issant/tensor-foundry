import numpy as np
import tensorfoundry.neural_networks.activation as activation
import tensorfoundry.neural_networks.optimizer as optimizers
import tensorfoundry.metrics as metrics
from .weight_initializer import WeightInitializer as WE

class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[tuple[int, activation.Base]], weight_initializer: WE = WE.RANDOM):
        self.layers = []

        if len(hidden_layers) == 0:
            self.layers.append((input_size, output_size, activation.Linear))
        else:
            self.layers.append((input_size, hidden_layers[0][0], activation.Linear))

            for i in range(1, len(hidden_layers)):
                self.layers.append((hidden_layers[i - 1][0], hidden_layers[i][0], hidden_layers[i - 1][1]))

            self.layers.append((hidden_layers[-1][0], output_size, hidden_layers[-1][1]))

        self.weights = weight_initializer.get_weights(self.layers)
        self.biases = [np.zeros((1, l[1])) for l in self.layers]

        self.d_weights = [None] * len(self.layers)
        self.d_biases = [None] * len(self.layers)

    def forward(self, input: np.ndarray):
        self.pres = []
        self.posts = [input]

        a = input
        for i in range(len(self.layers)):
            z = (a @ self.weights[i]) + self.biases[i]
            self.pres.append(z)

            a = self.layers[i][2].compute(z)
            self.posts.append(a)

        return a

    def backward(self, target: np.ndarray, loss_func: metrics.Base):
        num_layers = len(self.layers)
        m = target.shape[0]

        if self.layers[-1][2] == activation.Softmax and loss_func == metrics.CategoricalCrossEntropy:
            d_z = self.posts[-1] - target
        else:
            d_a = loss_func.derivative(target, self.posts[-1])
            d_z = d_a * self.layers[-1][2].derivative(self.pres[-1])

        self.d_weights[-1] = (self.posts[-2].T @ d_z) / m
        self.d_biases[-1] = np.sum(d_z, axis=0, keepdims=True) / m
        
        d_a = d_z @ self.weights[-1].T

        for i in range(num_layers - 2, -1, -1):
            d_z = d_a * self.layers[i][2].derivative(self.pres[i])

            self.d_weights[i] = (self.posts[i].T @ d_z) / m
            self.d_biases[i] = np.sum(d_z, axis=0, keepdims=True) / m

            if i > 0:
                d_a = d_z @ self.weights[i].T

    def update(self, learning_rate: float, optimizer: optimizers.Base):
        self.weights, self.biases = optimizer.step(self.weights, self.biases, self.d_weights, self.d_biases, learning_rate)

    def train_step(self, input: np.ndarray, target: np.ndarray, learning_rate: float, loss_func: metrics.Base, optimizer: optimizers.Base):
        self.forward(input)
        self.backward(target, loss_func)
        self.update(learning_rate, optimizer)

    def train(self, input: np.ndarray, target: np.ndarray, epochs: int, learning_rate: float, loss_func: metrics.Base, optimizer: optimizers.Base, batch_size: int = 32, log: bool = True):
        num_samples = input.shape[0]

        for epoch in range(epochs):
            if optimizer is optimizers.MiniBatchSGD:
                batches = optimizer.batch(input, target, num_samples, batch_size)
            else:
                batches = optimizer.batch(input, target, num_samples)

            for batch_input, batch_target in batches:
                self.train_step(batch_input, batch_target, learning_rate, loss_func, optimizer)
            
            if log:
                preds = self.forward(input)
                loss_val = np.mean(loss_func.compute(target, preds)).item()

                predicted_classes = np.argmax(preds, axis=1)
                target_classes = np.argmax(target, axis=1)
                accuracy = np.mean(predicted_classes == target_classes)

                print(f"Epoch {epoch:5d} | Overall Loss: {loss_val:.6f} | Accuracy: {accuracy:.6f}")

    def test(self, input: np.ndarray, target: np.ndarray, loss_func: metrics.Base):
        preds = self.forward(input)
        loss_val = np.mean(loss_func.compute(target, preds)).item()

        predicted_classes = np.argmax(preds, axis=1)
        target_classes = np.argmax(target, axis=1)
        accuracy = np.mean(predicted_classes == target_classes)
        
        print(f"Test Loss: {loss_val:.6f} | Test Accuracy: {accuracy:.6f}")
