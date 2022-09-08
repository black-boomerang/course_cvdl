import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod


class BaseLayer:
    """
    Абстрактный класс слоя. Слой хранит параметры, выполняет операции над данными и
    вычисляет градиенты параметров и входных данных по градиентам выходных данных.
    Тесты всех слоев используют только три метода ниже, их необходимо реализовать
    в наследниках этого класса.
    В этом абстрактном классе ничего не нужно изменять.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        # Здесь будут храниться тензоры-параметры.
        # Все тензоры-параметры должны быть созданы в конструкторе наследника и
        # помещены в этот список.
        self.parameters = []
        # Здесь хранятся тензоры-градиенты параметров после backward
        # Все тензоры-градиенты должны быть созданы в конструкторе наследника.
        self.parameters_grads = []

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Прямой проход, выполняющий операцию над входными данными.
        Слои принимают и возвращают только один тензор.
        """
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Обратный проход, принимающий градиент ошибки по выходному тензору,
         (результату метода .forward) и возвращающий градиент ошибки по
         входному тензору (аргумент метода .forward).
        Метод .backward всегда вызывается после одного вызова .forward.
        Метод .backward должен записать в .parameters_grads градиенты
         параметров.
        Слои принимают и возвращают только один тензор.
        """
        pass
