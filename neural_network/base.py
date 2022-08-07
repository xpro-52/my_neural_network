from typing import Callable, List
import numpy as np

from .layers import FullyConnectedLayer, Layer
from .optimizer import Optimizer


class BaseNN:
    def __init__(self, features_dim: int, random_seed: int) -> None:
        self.random = np.random.RandomState(random_seed)
        self.layers: List[Layer] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, error: np.ndarray) -> None:
        delta = error
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def parameters(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                parameters.append(layer.w)
                parameters.append(layer.theta)
        return parameters

    def gradients(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                dw, dtheta = layer.gradients()
                gradients.append(dw)
                gradients.append(dtheta)
        return gradients

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def __str__(self) -> str:
        return "<%s: {\n %s \n}>" % (
            self.__class__.__name__,
            "\n".join(["\t" + str(layer) + "," for layer in self.layers]),
        )


def learnNN(
    nn: BaseNN,
    train_X: np.ndarray,
    train_y: np.ndarray,
    rho: float,
    optimizer: Optimizer,
    batch_size: int,
    iteration: int,
    random_seed: int,
    process=False,
) -> List[np.float64]:

    train_accuracies = []
    org_random = np.random.RandomState(random_seed)
    random_seeds = org_random.randint(0, 1000, iteration)

    for i in range(iteration):
        random = np.random.RandomState(random_seeds[i])
        batch_indices = random.choice(train_X.shape[0], batch_size)
        y = nn.predict(train_X[batch_indices, :])
        delta = y - train_y[batch_indices]
        nn.backward(delta)
        optimizer.update(nn.parameters(), nn.gradients(), rho)

        accuracy = (
            np.sum(
                np.where(
                    y >= 0.5,
                    1,
                    0,
                )
                == train_y[batch_indices]
            )
            / y.size
        )
        train_accuracies.append(accuracy)
        if process and (i + 1) % 20 == 0:
            print("train accuracy: %f" % accuracy)

    optimizer.reset_parameters()
    return train_accuracies


# def learnNNWithoutBatch(  # deprecated
#     nn: BaseNN,
#     train_X: np.ndarray,
#     train_y: np.ndarray,
#     rho: float,
#     optimizer: Optimizer,
#     iteration: int,
# ) -> List[np.float64]:
#     train_accuracies = []
#     predictions = np.zeros_like(train_y)

#     for i in range(iteration):
#         for j in range(train_X.shape[0]):
#             prediction = nn.predict(train_X[j].reshape(1, -1))
#             predictions[j] = prediction
#             delta = prediction - train_y[j]
#             nn.backward(delta)
#             optimizer.update(nn.parameters(), nn.gradients(), rho)

#         accuracy = (
#             np.sum(np.where(predictions >= 0.5, 1, 0) == train_y)
#             / train_y.size
#         )
#         train_accuracies.append(accuracy)
#         if (i + 1) % 20 == 0:
#             print("train accuracy: %f" % accuracy)

#     return train_accuracies


def cross_check(
    nn: BaseNN,
    random_seed: int,
    X: np.ndarray,
    y: np.ndarray,
    rho: float,
    optimizer: Optimizer,
    batch_size: int,
    iteration: int,
    n: int,
    process=True,
):
    split_indices = np.array_split(np.arange(0, X.shape[0], 1), n)
    acc_list = np.zeros(n)
    print("--- Start cross check(n=%d) ---" % n)
    for i in range(n):
        train_X = np.delete(X, split_indices[i], axis=0)
        train_y = np.delete(y, split_indices[i]).reshape(-1, 1)
        test_X = X[split_indices[i]]
        test_y = y[split_indices[i]].reshape(-1, 1)

        learnNN(
            nn,
            train_X,
            train_y,
            rho,
            optimizer,
            batch_size,
            iteration,
            random_seed,
        )

        predictions = nn.predict(test_X)
        acc = (
            100
            * np.sum(np.where(predictions >= 0.5, 1, 0) == test_y)
            / test_y.size
        )
        acc_list[i] = acc
        if process:
            print("End of part %d: %.3f%%" % (i + 1, acc))
        nn.reset_parameters()
        optimizer.reset_parameters()

    return np.mean(acc_list), np.std(acc_list)
