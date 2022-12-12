import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class MetrixModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net_out_size = 14
        self.net_out_channels = 512

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.embed_layer = nn.Linear(self.net_out_channels, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.net_out_channels, self.net_out_size, self.net_out_size)
        x = self.avg_pool(x) + self.max_pool(x)
        x = x.squeeze()
        x = self.embed_layer(x)
        return F.normalize(x, dim=1)

    def get_features(self, x):
        x = self.backbone(x)
        return x.view(-1, self.net_out_channels, self.net_out_size, self.net_out_size)

    def get_embeddings_by_features(self, x):
        x = self.avg_pool(x) + self.max_pool(x)
        x = x.squeeze()
        x = self.embed_layer(x)
        return F.normalize(x, dim=1)
