import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert (in_channels > 0)
        assert (out_channels > 0)
        assert (kernel_size % 2 == 1)
        super().__init__()
        limit = np.sqrt(6 / (in_channels + out_channels))
        self.parameters.append(np.random.uniform(-limit, limit, size=(
            out_channels, in_channels, kernel_size, kernel_size)))
        self.parameters.append(np.zeros(out_channels))
        self.parameters_grads.append(
            np.zeros((out_channels, in_channels, kernel_size, kernel_size)))
        self.parameters_grads.append(np.zeros(out_channels))
        self.padding = self.kernel_size // 2
        self.pad_input = None

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pad_width = [[0, 0]] * len(tensor.shape)
        for i in axis:
            pad_width[i] = [one_side_pad, one_side_pad]
        return np.pad(tensor, pad_width, constant_values=0)

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        B, C, H, W = input.shape
        self.pad_input = self._pad_zeros(input, self.padding)
        result = np.zeros((B, self.out_channels, H, W))

        for i in range(H):
            for j in range(W):
                cur_window = self.pad_input[:, None, :, i:(i + self.kernel_size), j:(j + self.kernel_size)]
                result[:, :, i, j] = (cur_window * self.parameters[0]).sum((2, 3, 4)) + self.parameters[1]
        return result

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        B, _, H, W = output_grad.shape
        grad_input = np.zeros_like(self.pad_input)
        self.parameters_grads[0] = np.zeros_like(self.parameters[0])
        self.parameters_grads[1] = np.zeros_like(self.parameters[1])

        for channel in range(self.out_channels):
            for h in range(H):
                h_end = h + self.kernel_size
                for w in range(W):
                    w_end = w + self.kernel_size
                    cur_grad = output_grad[:, channel, h, w][:, None, None, None]
                    self.parameters_grads[0][channel] += (cur_grad * self.pad_input[:, :, h:h_end, w:w_end]).sum(axis=0)
                    grad_input[:, :, h:h_end, w:w_end] += (cur_grad * self.parameters[0][channel])
            self.parameters_grads[1][channel] += output_grad[:, channel].sum()

        return grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
