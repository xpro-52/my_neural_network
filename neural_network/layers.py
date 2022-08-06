import numpy as np
from typing import Optional, Tuple


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta

    def reset_parameters(self) -> None:
        return


class NeuronsLayer(Layer):
    pass


class ActivationFunctionLayer(Layer):
    def __init__(self) -> None:
        self.y: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta

    def reset_parameters(self) -> None:
        self.y = None

    def __str__(self) -> str:
        return "<%s>" % self.__class__.__name__


class Sigmoid(ActivationFunctionLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        lower = x < -709
        y[lower] = 0.0
        upper = x > 709
        y[upper] = 1.0
        mask = ~(lower | upper)
        y[mask] = 1 / (1 + np.exp(-x[mask]))
        self.y = y
        return y

    def backward(self, delta: np.ndarray) -> np.ndarray:
        if self.y is None:
            raise Exception()
        return delta * (1.0 - self.y) * self.y


class ReLU(ActivationFunctionLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0, x, 0)
        self.y = y
        return y

    def backward(self, delta: np.ndarray) -> np.ndarray:
        if not self.y:
            raise Exception()
        return delta * np.where(self.y > 0, 1, 0)


class LeakyReLU(ActivationFunctionLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0, x, 0.01 * x)
        self.y = y
        return y

    def backward(self, delta: np.ndarray) -> np.ndarray:
        if not self.y:
            raise Exception()
        return delta * np.where(self.y > 0, 1, 0.01)


class FullyConnectedLayer(NeuronsLayer):
    def __init__(
        self,
        w: np.ndarray,
        theta: np.ndarray,
    ) -> None:
        self.x: Optional[np.ndarray] = None
        self.org_w = w.copy()
        self.org_theta = theta.copy()
        self.w = w
        self.theta = theta
        self.delta: Optional[np.ndarray] = None

    def gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.delta is None or self.x is None:
            raise Exception()
        return self.x.T @ self.delta, -np.sum(self.delta, axis=0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        y = self.x @ self.w - self.theta
        return y

    def backward(self, delta: np.ndarray) -> np.ndarray:
        self.delta = delta
        return self.delta @ self.w.T

    def reset_parameters(self) -> None:
        self.x = None
        self.w = self.org_w
        self.theta = self.org_theta
        self.delta = None

    def __str__(self) -> str:
        return "<%s: w%s, theta%s>" % (
            self.__class__.__name__,
            self.w.shape,
            self.theta.shape,
        )
