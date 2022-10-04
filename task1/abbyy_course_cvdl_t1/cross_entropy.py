import numpy as np
from .base import BaseLayer


class CrossEntropyLoss(BaseLayer):
    """
    Слой-функция потерь, категориальная кросс-энтропия для задачи класификации на
    N классов.
    Применяет softmax к входным данных.
    """

    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Принимает два тензора - предсказанные логиты классов и правильные классы.
        Prediction и target имеют одинаковый размер вида
         [B, C, 1, ... 1] т.е. имеют 2 оси (batch size и channels) длины больше 1
          и любое количество осей единичной длины.
        В predictions находятся логиты, т.е. к ним должен применяться softmax перед вычислением
         энтропии.
        В target[B, C, (1, ..., 1)] находится 1, если объект B принадлежит классу C, иначе 0 (one-hot представление).
        Возвращает np.array[B] c лоссом по каждому объекту в мини-батче.

        P[B, c] = exp(pred[B, c]) / Sum[c](exp(pred[B, c])
        Loss[B] = - Sum[c]log( prob[B, C] * target[B, C]) ) = -log(prob[B, C_correct])
        """
        self.pred = pred
        self.target = target
        prob = np.exp(pred - pred.max(axis=1, keepdims=True))
        prob = prob / prob.sum(axis=1, keepdims=True)
        return -np.log((prob * target).sum(axis=1))

    def backward(self) -> np.ndarray:
        """
        Возвращает градиент лосса по pred, т.е. первому аргументу .forward
        Не принимает никакого градиента по определению.
        """
        prob = np.exp(self.pred - self.pred.max(axis=1, keepdims=True))
        prob = prob / prob.sum(axis=1, keepdims=True)
        return prob - self.target
