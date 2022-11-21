import json
import os
from typing import Dict
from urllib.request import urlretrieve

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed

from dataset import CocoTextDataset


def load_model(model_name: str, weights_dict: Dict[str, Dict]) -> str:
    weights_dir = 'weights'
    dist_path = os.path.join(weights_dir, f'{model_name}.pth')
    if os.path.exists(dist_path):
        return dist_path

    mmcv.mkdir_or_exist(weights_dir)
    urlretrieve(weights_dict[model_name]['weights_url'], dist_path)
    return dist_path


def modify_config(cfg: Config, weights_path: str, data_path: str) -> Config:
    cfg.dataset_type = 'CocoTextDataset'
    cfg.data_root = data_path

    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 4
    cfg.data.train.dataset.type = cfg.data.val.type = cfg.dataset_type
    cfg.data.train.dataset.data_root = cfg.data.val.data_root = cfg.data_root
    # cfg.data.train.classes = cfg.data.val.classes = ('text',)
    cfg.data.train.dataset.ann_file = 'cocotext.train.json'
    cfg.data.val.ann_file = 'cocotext.val.json'
    cfg.data.train.dataset.img_prefix = cfg.data.val.img_prefix = 'train2014'

    cfg.model.input_size = (640, 640)
    cfg.model.bbox_head.num_classes = 1

    cfg.load_from = weights_path
    cfg.optimizer.lr = 0.0008 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 100
    cfg.log_config.num_last_epochs = 0

    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    cfg.work_dir = fr'workdirs\{model_name}'
    cfg.evaluation.metric = 'bbox'
    cfg.evaluation.save_best = 'auto'
    cfg.evaluation.interval = 1
    cfg.checkpoint_config.interval = 1
    cfg.gpu_ids = [0]
    cfg.device = 'cuda'
    cfg.runner.max_epochs = 15
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
    cfg.seed = 12
    set_random_seed(12, deterministic=False)

    return cfg


def load_config(model_name: str, weights_urls_path: str = 'models_info.json', data_path: str = r'.\data',
                pretrained: str = None) -> Config:
    configs_dir = 'configs'
    config_path = os.path.join(configs_dir, f'{model_name}.py')

    with open(weights_urls_path, 'r') as f:
        models_dict = json.load(f)

    if not os.path.exists(config_path):
        mmcv.mkdir_or_exist(configs_dir)
        config_url = models_dict[model_name]['config_path']
        urlretrieve(f'https://raw.githubusercontent.com/open-mmlab/mmdetection/master/{config_url}', config_path)

    if pretrained is not None:
        weights_path = pretrained
    else:
        weights_path = load_model(model_name, models_dict)

    cfg = Config.fromfile(config_path)
    cfg = modify_config(cfg, os.path.abspath(weights_path), os.path.abspath(data_path))
    return cfg
