import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        limit = np.sqrt(6 / (in_features + out_features))
        self.parameters.append(
            np.random.uniform(-limit, limit, size=(out_features, in_features)))
        self.parameters.append(np.zeros(out_features))
        self.parameters_grads = [None, None]
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = input @ self.parameters[0].T + self.parameters[1]
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        if output_grad.ndim > 2:
            output_grad_t = output_grad.transpose(*tuple(range(output_grad.ndim - 2)), -1, -2)
        else:
            output_grad_t = output_grad.T
        self.parameters_grads[0] = (output_grad_t @ self.input).sum(axis=tuple(range(output_grad.ndim - 2)))
        self.parameters_grads[1] = (np.ones(output_grad.shape[-2]) @ output_grad).sum(axis=tuple(range(output_grad.ndim - 2)))
        return output_grad @ self.parameters[0]
