import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

pd.options.mode.chained_assignment = None


class ClothesDataset(Dataset):
    def __init__(self, root, mode='train', transform=None) -> None:
        self.root = root
        meta = pd.read_csv(os.path.join(root, 'list_eval_partition.txt'), skiprows=1, sep='\s+')
        self.meta = meta[meta['evaluation_status'] == mode].reset_index(drop=True)
        self.meta['item_id'] = self.meta['item_id'].rank(method='dense').astype(int)
        self.train_mode = (mode == 'train')
        self.transform = transform

    def __getitem__(self, index):
        path = os.path.join(self.root, self.meta.loc[index, 'image_name'])
        # img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.train_mode:
            return img, self.meta.loc[index, 'item_id']
        return img

    def __len__(self) -> int:
        return len(self.meta)

    def get_meta(self):
        return self.meta
