import numpy as np
import pandas as pd


def recall_top_k(preds: np.ndarray, test_data: pd.DataFrame, query_data: pd.DataFrame, k: int = 1):
    metrics_data = query_data.copy()

    items_count = test_data.groupby('item_id').count()['image_name']
    metrics_data['rel_count'] = items_count.loc[query_data['item_id']].values
    metrics_data['k'] = k
    metrics_data['rel_count'] = metrics_data[['rel_count', 'k']].min(axis=1)

    pred_items = test_data.loc[preds.ravel(), 'item_id'].values.reshape(*preds.shape)
    metrics_data['pred_rel_count'] = (pred_items == query_data['item_id'][:, None]).sum(axis=1)
    return (metrics_data['pred_rel_count'] / metrics_data['rel_count']).mean()


def article_top_k(preds: np.ndarray, test_data: pd.DataFrame, query_data: pd.DataFrame, k: int = 1):
    metrics_data = query_data.copy()

    items_count = test_data.groupby('item_id').count()['image_name']
    metrics_data['rel_count'] = items_count.loc[query_data['item_id']].values
    metrics_data['k'] = k
    metrics_data['rel_count'] = metrics_data[['rel_count', 'k']].min(axis=1)

    pred_items = test_data.loc[preds.ravel(), 'item_id'].values.reshape(*preds.shape)
    metrics_data['pred_rel_count'] = (pred_items == query_data['item_id'][:, None]).sum(axis=1)
    return ((metrics_data['pred_rel_count'] >= 1) / 1).mean()
