import datetime
import gc
import os
import time
from collections import defaultdict
from typing import List, Dict, Union, Callable

import numpy as np
import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_pairs_indices
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from task_project.dataset import ClothesDataset
from task_project.losses import MixupContrastiveLoss
from task_project.metrics import recall_top_k
from task_project.mixup import get_mixup_pos_neg_triplets
from task_project.models import MetrixModel
from task_project.utils import init_datasets, compute_distances


def show_plot(history: defaultdict, elapsed_time: int, epoch: int):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train']['iters'], history['train']['loss'], label='train')
    plt.ylabel('Лосс', fontsize=15)
    plt.xlabel('Итерация', fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val']['iters'], history['val']['recall'], label='val')
    plt.ylabel('R@1', fontsize=15)
    plt.xlabel('Итерация', fontsize=15)
    plt.legend()

    plt.suptitle(f'Итерация {history["train"]["iters"][-1]}, эпоха: {epoch}, время: '
                 f'{datetime.timedelta(seconds=elapsed_time)}, лосс: {history["train"]["loss"][-1]:.3f}', fontsize=15)

    plt.show()


def validate(test_dataset: ClothesDataset, query_dataset: ClothesDataset, model: nn.Module,
             k: Union[List[int], int] = 1, metric: Callable = recall_top_k) -> Union[Dict[int, float], float]:
    model.eval()
    distances = compute_distances(model, test_dataset, query_dataset)
    # best_images = distances.argmin(axis=1, keepdims=True)

    return_int = isinstance(k, int)
    if return_int:
        k = [k]

    recall = {}
    for i in k:
        best_images = np.argpartition(distances, i, axis=1)[:, :i]
        indices = np.repeat(np.arange(distances.shape[0])[:, None], i, axis=1)
        sorted_images = np.argsort(distances[indices, best_images], axis=1)[:, :i]
        best_images = best_images[indices[:, :i], sorted_images]
        recall[i] = metric(best_images, test_dataset.get_meta(), query_dataset.get_meta(), k=i)

    if return_int:
        return recall[k]

    return recall


def train(root: str, lr: float, epochs: int, checkpoint_path: str, lr_decay_rate: float = 0.25, embed_dim: int = 512,
          pretrained_path: str = None, start_epoch: int = 0, mixup_mode: bool = True):
    gc.collect()
    log_step = 1

    train_dataset, test_dataset, query_dataset = init_datasets(root)
    sampler = MPerClassSampler(train_dataset.get_meta()['item_id'], 5, config.batch_size,
                               length_before_new_iter=len(train_dataset))
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=config.batch_size,
                                  num_workers=6, pin_memory=True)
    batch_per_epoch = len(train_dataloader)

    model = MetrixModel(embed_dim).to(config.device)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=config.device), strict=False)

    distance = LpDistance()
    loss_fn = ContrastiveLoss(pos_margin=0, neg_margin=0.5, distance=distance)
    mixup_loss_fn = MixupContrastiveLoss(pos_margin=0, neg_margin=0.5, distance=distance)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.StepLR(optimizer, len(train_dataloader), gamma=lr_decay_rate)

    # miner = BatchEasyHardMiner(pos_strategy='easy', neg_strategy='semihard', distance=CosineSimilarity())

    iteration = 0
    history = defaultdict(lambda: defaultdict(list))
    losses = []
    os.makedirs(checkpoint_path, exist_ok=True)
    start_time = time.time()
    for epoch in range(epochs):
        if epoch < start_epoch:
            for _ in range(len(train_dataloader)):
                optimizer.step()
                scheduler.step()
            continue

        model.train()

        for i, (x, y) in tqdm(enumerate(train_dataloader), total=batch_per_epoch):
            optimizer.zero_grad()

            x = x.to(config.device)
            y = y.to(config.device)

            embeddings = model(x)
            loss = loss_fn(embeddings, y)

            if mixup_mode:
                features = model.get_features(x)
                a1, pos, a2, neg = get_all_pairs_indices(y)
                anc, pos, neg = get_mixup_pos_neg_triplets(a1, pos, a2, neg, embeddings, distance, 3)
                lambd = np.random.beta(2, 2)
                anchor_emb = model.get_embeddings_by_features(features[anc])
                mixup_emb = model.get_embeddings_by_features(lambd * features[pos] + (1 - lambd) * features[neg])
                loss += 0.4 * mixup_loss_fn(anchor_emb, mixup_emb, lambd)

            if loss.isfinite().all():
                loss.backward()

                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
            iteration += 1

        history['train']['loss'].append(np.mean(losses))
        history['train']['iters'].append(iteration)
        losses = []

        if epoch % log_step == log_step - 1:
            recall = validate(test_dataset, query_dataset, model)
            history['val']['recall'].append(recall)
            history['val']['iters'].append(iteration)

        clear_output()
        show_plot(history, int(time.time() - start_time), epoch)

        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch{epoch}.pt'))

    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'latest.pt'))

    clear_output()
    show_plot(history, int(time.time() - start_time), epochs)

    return model
