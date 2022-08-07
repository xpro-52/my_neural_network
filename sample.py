import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from neural_network.layers import FullyConnectedLayer, Sigmoid
from neural_network.base import BaseNN, learnNN
from neural_network.optimizer import Adam, StochasticGradientDecent
from neural_network.layers import LeakyReLU


class ThreeLayerNN(BaseNN):
    def __init__(self, features_dim: int, random_seed: int) -> None:
        super().__init__(features_dim, random_seed)
        self.layers = [
            FullyConnectedLayer(
                self.random.randn(features_dim, 100), self.random.randn(100)
            ),
            Sigmoid(),
            FullyConnectedLayer(
                self.random.randn(100, 50), self.random.randn(50)
            ),
            Sigmoid(),
            FullyConnectedLayer(
                self.random.randn(50, 1), self.random.randn(1)
            ),
            Sigmoid(),
        ]


def load_sample_data():
    random_seed = 21
    random = np.random.RandomState(random_seed)
    x = random.uniform(-3, 3, size=600).reshape(-1, 1)
    y = np.sin(x)
    y = np.where(y >= 0.5, 1, 0).reshape(-1, 1)
    train_X = x[:500]
    train_y = y[:500]
    test_X = x[500:]
    test_y = y[500:]
    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_sample_data()

    random_seed = 21
    nn = ThreeLayerNN(train_X.shape[1], random_seed)
    print(nn)

    rho = 1e-3  # learning rate
    # optimizer = StochasticGradientDecent()
    optimizer = Adam()
    batch_size = 100
    iteration = 500

    train_accuracies = learnNN(
        nn,
        train_X,
        train_y,
        rho,
        optimizer,
        batch_size,
        iteration,
        random_seed,
        process=True,  # print training log
    )

    plt.plot(range(iteration), train_accuracies)
    min_acc = min(train_accuracies)
    plt.yticks(np.arange(min_acc - min_acc % 0.025, 1.025, 0.025))
    plt.title("training accuracy")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.savefig("./nn.png")

    predictions = np.zeros_like(test_y)
    predictions = np.where(nn.predict(test_X) >= 0.5, 1, 0)
    print(
        "test acc: %.3f" % (np.sum(predictions == test_y) / predictions.size)
    )
    nn.reset_parameters()
