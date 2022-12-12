import torch


def get_mixup_pos_neg_triplets(a1, pos, a2, neg, embeddings, distance, k=3):
    new_anc = []
    new_pos = []
    new_neg = []

    distances = distance(embeddings)
    for a1_ind, anchor in enumerate(a1):
        a2_ind = torch.where(a2 == anchor)[0]
        neg_distances = distances[a2[a2_ind], neg[a2_ind]]
        a2_ind = torch.sort(neg_distances)[1][:k]
        new_anc.extend(a2[a2_ind].tolist())
        new_pos.extend([pos[a1_ind].item()] * k)
        new_neg.extend(neg[a2_ind].tolist())

    new_anc = torch.LongTensor(new_anc)
    new_pos = torch.LongTensor(new_pos)
    new_neg = torch.LongTensor(new_neg)
    return new_anc, new_pos, new_neg
