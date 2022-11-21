import json

import requests
import yaml
from tqdm.auto import tqdm

from settings import *


def iter_models_meta():
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    configs_url = 'https://api.github.com/repos/open-mmlab/mmdetection/contents/configs'
    config_urls = [content_json['url'] for content_json in requests.get(configs_url).json()]
    for config_url in tqdm(config_urls):
        content_json = requests.get(config_url, headers=headers).json()
        filenames = {c['name']: i for i, c in enumerate(content_json)}
        if 'metafile.yml' in filenames:
            meta_index = filenames['metafile.yml']
            yield yaml.safe_load(requests.get(content_json[meta_index]['download_url']).text)


def get_models_info_file(filename: str) -> None:
    models_dict = {}
    for model_meta in iter_models_meta():
        for i in range(len(model_meta['Models'])):
            if 'Weights' in model_meta['Models'][i] and 'Config' in model_meta['Models'][i]:
                weights_url = model_meta['Models'][i]['Weights']
                config_path = model_meta['Models'][i]['Config']
                models_dict[weights_url.split('/')[-2]] = {
                    'weights_url': weights_url,
                    'config_path': config_path
                }

    with open(filename, 'w') as f:
        json.dump(models_dict, f)


def convert_anns_to_suit_format(src_path: str, dest_train_path: str, dest_val_path: str) -> None:
    with open(src_path, 'r') as f:
        src_anns_data = json.loads(f.read())

    dest_anns_train_data = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'text'}]}
    dest_anns_val_data = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'text'}]}
    img_set = {}
    for id, img_info in src_anns_data['imgs'].items():
        image = {
            'file_name': img_info['file_name'],
            'height': img_info['height'],
            'width': img_info['width'],
            'id': int(id)
        }
        img_set[int(id)] = img_info['set']
        if img_info['set'] == 'train':
            dest_anns_train_data['images'].append(image)
        elif img_info['set'] == 'val':
            dest_anns_val_data['images'].append(image)

    for id, ann_info in src_anns_data['anns'].items():
        annotation = {
            'area': ann_info['area'],
            'iscrowd': 0,
            'image_id': ann_info['image_id'],
            'bbox': ann_info['bbox'],
            'category_id': 0,
            'id': int(id)
        }
        if img_set[ann_info['image_id']] == 'train':
            dest_anns_train_data['annotations'].append(annotation)
        elif img_set[ann_info['image_id']] == 'val':
            dest_anns_val_data['annotations'].append(annotation)

    with open(dest_train_path, 'w') as f:
        json.dump(dest_anns_train_data, f)

    with open(dest_val_path, 'w') as f:
        json.dump(dest_anns_val_data, f)
