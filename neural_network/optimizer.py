from typing import List, Optional
import numpy as np


EPSILON = 1e-7


class Optimizer:
    def update(
        self,
        parameters: List[np.ndarray],
        gradients: List[np.ndarray],
        rho: float,
        *args,
        **kwargs
    ) -> None:
        return

    def reset_parameters(self) -> None:
        return

    def __str__(self) -> str:
        return "<%s>" % self.__class__.__name__


class StochasticGradientDecent(Optimizer):
    def update(
        self,
        parameters: List[np.ndarray],
        gradients: List[np.ndarray],
        rho: float,
        *args,
        **kwargs
    ) -> None:
        for parameter, gradient in zip(parameters, gradients):
            parameter -= rho * gradient
        return


class Momentum(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self.v: Optional[List[np.ndarray]] = None

    def update(
        self,
        parameters: List[np.ndarray],
        gradients: List[np.ndarray],
        rho: float,
        mu=0.9,
        *args,
        **kwargs
    ) -> None:
        if not self.v:
            self.v = []
            for parameter in parameters:
                self.v.append(np.zeros_like(parameter))

        for i, gradient in enumerate(gradients):
            self.v[i] += mu * self.v[i] - rho * gradient
            parameters[i] += self.v[i]
        return

    def reset_parameters(self) -> None:
        self.v = None
        return


class AdaGrad(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self.h: Optional[List[np.ndarray]] = None

    def update(
        self,
        parameters: List[np.ndarray],
        gradients: List[np.ndarray],
        rho: float,
        *args,
        **kwargs
    ) -> None:
        if not self.h:
            self.h = []
            for parameter in parameters:
                self.h.append(np.zeros_like(parameter))

        for i, gradient in enumerate(gradients):
            self.h[i] += gradient * gradient
            parameters[i] -= rho * gradient / (np.sqrt(self.h[i]) + EPSILON)
        return

    def reset_parameters(self) -> None:
        self.h = None
        return


class Adam(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self.t = 0  # time step
        self.m: Optional[List[np.ndarray]] = None
        self.v: Optional[List[np.ndarray]] = None

    def update(
        self,
        parameters: List[np.ndarray],
        gradients: List[np.ndarray],
        rho: float,
        beta1=0.9,
        beta2=0.999,
        *args,
        **kwargs
    ) -> None:
        if not self.m:
            self.m = []
            self.v = []
            for parameter in parameters:
                self.m.append(np.zeros_like(parameter))
                self.v.append(np.zeros_like(parameter))

        self.t += 1
        alpha_t = rho * np.sqrt(1 - beta2**self.t) / (1 - beta1**self.t)

        if not self.v:
            raise Exception()

        for i, gradient in enumerate(gradients):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradient
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * gradient**2
            parameters[i] -= (
                alpha_t
                * (self.m[i] / (1 - beta1**self.t))
                / (np.sqrt(self.v[i] / (1 - beta2**self.t)) + EPSILON)
            )
        return

    def reset_parameters(self) -> None:
        self.t = 0
        self.m = None
        self.v = None
        return
