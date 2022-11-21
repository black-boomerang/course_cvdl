import json
import os
from typing import List

import numpy as np
from mmdet.apis import init_detector, inference_detector
from tqdm.auto import tqdm

from abbyy_course_cvdl_t3.utils import dump_detections_to_cocotext_json
from config import load_config


def save_results(results: List[np.ndarray], image_ids: List[int], pred_path: str) -> None:
    results = np.vstack(results)
    xlefts = results[:, 0].astype(int)
    ytops = results[:, 1].astype(int)
    widths = (results[:, 2] - xlefts).astype(int)
    heights = (results[:, 3] - ytops).astype(int)
    scores = results[:, 4]

    pred_dir = os.path.dirname(pred_path)
    os.makedirs(pred_dir, exist_ok=True)

    dump_detections_to_cocotext_json(
        image_ids=image_ids,
        xlefts=xlefts.tolist(),
        ytops=ytops.tolist(),
        widths=widths.tolist(),
        heights=heights.tolist(),
        scores=scores.tolist(),
        path=pred_path
    )


def infer(model_type: str, data_dir: str, pred_path: str = 'predictions.json', device: str = 'cuda') -> None:
    cfg = load_config(model_type, data_path=data_dir)
    model = init_detector(cfg, fr'workdirs\{model_type}\latest.pth', device=device)

    with open(os.path.join(data_dir, 'cocotext.val.json'), 'r') as f:
        images = json.load(f)['images']

    image_ids = []
    results = []
    for image in tqdm(images):
        image_result = inference_detector(model, os.path.join(data_dir, 'train2014', image['file_name']))
        if len(image_result[0]) > 0:
            results.append(image_result[0])
        image_ids.extend([image['id']] * len(image_result[0]))

    save_results(results, image_ids, pred_path)
