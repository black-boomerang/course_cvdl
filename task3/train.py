import os

import mmcv
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def train(cfg: mmcv.Config):
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=False, validate=True)
