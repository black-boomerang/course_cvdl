import os
from typing import List

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

import config
from task_project.metrics import article_top_k
from task_project.train import validate
from task_project.utils import init_datasets, compute_distances, get_embeddings


def show_img(path: str):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')


def show_results(model: nn.Module, data_dir: str):
    model.eval()

    _, test_dataset, query_dataset = init_datasets(data_dir)
    test_meta = test_dataset.get_meta()
    query_meta = query_dataset.get_meta()

    distances = compute_distances(model, test_dataset, query_dataset)
    sorted_distances_ind = distances.argsort(axis=1)
    random_indexes = np.random.choice(len(query_dataset), 6, replace=False)

    plt.figure(figsize=(18, 22))
    for i, random_index in enumerate(random_indexes):
        plt.subplot(6, 6, i * 6 + 1)
        show_img(os.path.join(data_dir, query_meta.loc[random_index, 'image_name']))

        for j, close_ind in enumerate(sorted_distances_ind[random_index][:5]):
            plt.subplot(6, 6, i * 6 + j + 2)
            show_img(os.path.join(data_dir, test_meta.loc[close_ind, 'image_name']))
            plt.title(f'distance = {distances[random_index][close_ind]:.3f}')

    plt.tight_layout()
    plt.show()


def find_global_classes(meta: pd.DataFrame):
    name_split = meta['image_name'].str.split('/', expand=True)
    classes = name_split[2] + '(' + name_split[1] + ')'
    ranks = classes.rank(method='dense').astype(int).tolist()
    return ranks


def visualize_embeddings_2d(model: nn.Module, data_dir: str):
    _, dataset, _ = init_datasets(data_dir)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    embeddings = get_embeddings(model, dataloader)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    classes = find_global_classes(dataset.get_meta())
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=classes, cmap='jet')
    plt.legend()
    plt.title('Визуализация эмбеддингов в 2D')
    plt.legend()
    plt.show()


def compute_metrics(model: nn.Module, data_dir: str, k: List):
    model.eval()
    _, test_dataset, query_dataset = init_datasets(data_dir)
    recall = validate(test_dataset, query_dataset, model, k=k)
    article_metric = validate(test_dataset, query_dataset, model, k=k, metric=article_top_k)
    return recall, article_metric
