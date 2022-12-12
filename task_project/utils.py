import numpy as np
from sklearn.metrics import euclidean_distances
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm.auto import tqdm

import config
from task_project.dataset import ClothesDataset


def init_datasets(root: str):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = ClothesDataset(root, mode='train', transform=train_transform)
    test_dataset = ClothesDataset(root, mode='gallery', transform=transform)
    query_dataset = ClothesDataset(root, mode='query', transform=transform)

    return train_dataset, test_dataset, query_dataset


def get_embeddings(model: nn.Module, dataloader: DataLoader):
    embeds = []
    for X in tqdm(dataloader):
        X = X.to(config.device)
        embed = model(X).detach().cpu().numpy()
        embeds.extend(embed)
    embeds = np.array(embeds)
    return embeds


def compute_distances(model: nn.Module, test_dataset: Dataset, query_dataset: Dataset):
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=4,
                                 pin_memory=True)
    query_dataloader = DataLoader(query_dataset, shuffle=False, batch_size=config.batch_size, num_workers=4,
                                  pin_memory=True)

    print('Getting embeddings...')
    test_embeds = get_embeddings(model, test_dataloader)
    query_embeds = get_embeddings(model, query_dataloader)

    print('Computing distances...')
    distances = euclidean_distances(query_embeds, test_embeds)
    return distances
