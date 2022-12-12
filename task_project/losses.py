import torch.nn.functional as F
from pytorch_metric_learning.distances import LpDistance
from torch import nn


class MixupLoss:
    def __init__(self, ro_pos, ro_neg, sigma_pos, sigma_neg, tau, distance=LpDistance()):
        self.distance = distance
        self.ro_pos = ro_pos
        self.ro_neg = ro_neg
        self.sigma_pos = sigma_pos
        self.sigma_neg = sigma_neg
        self.tau = tau

    def __call__(self, anchor_emb, mixup_emb, lambd):
        distances = self.distance.pairwise_distance(anchor_emb, mixup_emb)
        pos_loss = self.sigma_pos(lambd * self.ro_pos(distances))
        neg_loss = self.sigma_neg((1 - lambd) * self.ro_neg(distances))
        loss = self.tau(pos_loss + neg_loss)
        return loss.mean()


class MixupContrastiveLoss(MixupLoss):
    def __init__(self, pos_margin=0, neg_margin=0.5, distance=LpDistance()):
        super().__init__(
            ro_pos=lambda x: F.relu(x - pos_margin),
            ro_neg=lambda x: F.relu(neg_margin - x),
            sigma_pos=nn.Identity(),
            sigma_neg=nn.Identity(),
            tau=nn.Identity(),
            distance=distance,
        )
