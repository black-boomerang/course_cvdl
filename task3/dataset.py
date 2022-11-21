from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class CocoTextDataset(CocoDataset):
    CLASSES = ('text',)

    PALETTE = [(220, 20, 60), ]
