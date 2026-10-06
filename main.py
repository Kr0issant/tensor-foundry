def main():
    import tensorfoundry.neural_networks.activation as activation
    import tensorfoundry.neural_networks.optimizer as optimizer
    import tensorfoundry.metrics as metrics
    from tensorfoundry.neural_networks.nn import NeuralNetwork, WE
    from loader import load_mnist_images, load_mnist_labels
    import numpy as np

    num_classes = 10

    X_train = load_mnist_images("dataset/train-images-idx3-ubyte/train-images-idx3-ubyte")
    y_train_raw = load_mnist_labels("dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
    y_train = np.eye(num_classes)[y_train_raw]

    X_test = load_mnist_images("dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
    y_test_raw = load_mnist_labels("dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
    y_test = np.eye(num_classes)[y_test_raw]


    input_size = 28 * 28
    hidden_layers = [
        # (256, Activation.SIGMOID),
        (128, activation.Sigmoid),
        (32, activation.Softmax)
    ]
    output_size = 10

    epochs = 16
    learning_rate = 1e-1
    loss_func = metrics.CategoricalCrossEntropy
    optimizer = optimizer.MiniBatchSGD
    batch_size = 64

    nn = NeuralNetwork(input_size, output_size, hidden_layers, WE.XAVIER)
    nn.train(X_train, y_train, epochs, learning_rate, loss_func, optimizer, batch_size)

    nn.test(X_test, y_test, loss_func)


if __name__ == "__main__":
    main()
