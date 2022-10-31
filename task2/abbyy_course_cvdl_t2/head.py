""" Здесь находится 'Голова' CenterNet, описана в разделе 4 статьи https://arxiv.org/pdf/1904.07850.pdf"""
import torch
from torch import nn
from torch.nn import functional as F


class CenterNetHead(nn.Module):
    """
    Принимает на вход тензор из Backbone input[B, K, W/R, H/R], где
    - B = batch_size
    - K = количество каналов (в ResnetBackbone K = 64)
    - H, W = размеры изображения на вход Backbone
    - R = output stride, т.е. во сколько раз featuremap меньше, чем исходное изображение
      (в ResnetBackbone R = 4)

    Возвращает тензора [B, C+4, W/R, H/R]:
    - первые C каналов: probs[B, С, W/R, H/R] - вероятности от 0 до 1
    - еще 2 канала: offset[B, 2, W/R, H/R] - поправки координат в пикселях от 0 до 1
    - еще 2 канала: sizes[B, 2, W/R, H/R] - размеры объекта в пикселях
    """

    def __init__(self, k_in_channels=64, c_classes: int = 2):
        super().__init__()
        self.c_classes = c_classes
        self.conv1 = nn.Conv2d(k_in_channels, k_in_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv_cls = nn.Conv2d(k_in_channels, c_classes, kernel_size=1, padding=0)
        self.conv_off = nn.Conv2d(k_in_channels, 2, kernel_size=1, padding=0)
        self.conv_size = nn.Conv2d(k_in_channels, 2, kernel_size=1, padding=0)

    def forward(self, input_t: torch.Tensor):
        x = self.act(self.conv1(input_t))
        class_heatmap = torch.sigmoid(self.conv_cls(x))
        offset_map = torch.sigmoid(self.conv_off(x))
        size_map = self.conv_size(x)
        return torch.cat([class_heatmap, offset_map, size_map], dim=1)
