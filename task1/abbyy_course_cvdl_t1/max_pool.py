import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """

    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert (kernel_size % 2 == 1)
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad_input = None
        self.result_side = None

    @staticmethod
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pad_width = [[0, 0]] * len(tensor.shape)
        for i in axis:
            pad_width[i] = [one_size_pad, one_size_pad]
        return np.pad(tensor, pad_width, constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[
                    -1] + 2 * self.padding - self.kernel_size) % self.stride == 0

        B, C, H, W = input.shape
        self.result_side = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        result = np.zeros((B, C, self.result_side, self.result_side))
        self.pad_input = self._pad_neg_inf(input, self.padding)

        in_h_index = 0
        for h in range(self.result_side):
            in_w_index = 0
            for w in range(self.result_side):
                cur_window = self.pad_input[:, :,
                             in_h_index:(in_h_index + self.kernel_size),
                             in_w_index:(in_w_index + self.kernel_size)]
                result[:, :, h, w] = np.amax(cur_window, axis=(-2, -1))
                in_w_index += self.stride
            in_h_index += self.stride
        return result

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(self.pad_input)
        B, C, _, _ = output_grad.shape
        channels, batches = np.meshgrid(np.arange(C), np.arange(B))

        in_h_index = 0
        for h in range(self.result_side):
            in_w_index = 0
            for w in range(self.result_side):
                cur_window = self.pad_input[:, :,
                             in_h_index:(in_h_index + self.kernel_size),
                             in_w_index:(in_w_index + self.kernel_size)]
                max_idx = np.argmax(cur_window.reshape(B, C, -1), axis=-1)
                max_h_idx, max_w_idx = np.unravel_index(max_idx, (self.kernel_size, self.kernel_size))
                max_h_idx = np.array(max_h_idx) + in_h_index
                max_w_idx = np.array(max_w_idx) + in_w_index
                grad[batches, channels, max_h_idx, max_w_idx] += output_grad[:, :, h, w]
                in_w_index += self.stride
            in_h_index += self.stride
        return grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
