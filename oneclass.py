import torch
import numpy as np
from sklearn.metrics import classification_report

def One_Class_GNN_prediction(center, radius, learned_representations, G, val_test, dic):

    for node in G.nodes:
        G.nodes[node]['embedding_opencast'] = learned_representations[node]

    interest = []
    outlier = []
    for node in G.nodes:
        if G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 1:
            interest.append(G.nodes[node]['embedding_opencast'])
        elif G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 0:
            outlier.append(G.nodes[node]['embedding_opencast'])

    interest = torch.stack(interest)
    outlier = torch.stack(outlier)
    
    dist_int = torch.sum((interest - center) ** 2, dim=1)

    scores_int = dist_int - radius ** 2

    dist_out = torch.sum((outlier - center) ** 2, dim=1)

    scores_out = dist_out - radius ** 2

    preds_interest = [1 if score < 0 else -1 for score in scores_int]
    preds_outliers = [-1 if score > 0 else 1 for score in scores_out]

    y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
    y_pred = list(preds_interest) + list(preds_outliers)
    if dic:
        return classification_report(y_true, y_pred, output_dict=dic)
    else:
        return print(classification_report(y_true, y_pred))

def one_class_loss(center, radius, learned_representations, mask):

    scores = anomaly_score(center, radius, learned_representations, mask)

    loss = torch.mean(torch.where(scores > 0, scores + 1, torch.exp(scores)))

    return loss

def anomaly_score(center, radius, learned_representations, mask):

    l_r_mask = torch.BoolTensor(mask)

    dist = torch.sum((learned_representations[l_r_mask] - center) ** 2, dim=1)

    scores = dist - radius ** 2

    return scores


def one_class_masking(G, train_val):
    train_mask = np.zeros(len(G.nodes), dtype='bool')
    unsup_mask = np.zeros(len(G.nodes), dtype='bool')

    normal_train_idx = []
    unsup_idx = []
    count = 0
    for node in G.nodes:
        if train_val:
            if G.nodes[node]['train'] == 1 or (G.nodes[node]['val'] == 1 and G.nodes[node]['label'] == 1):
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1
        else:
            if G.nodes[node]['train'] == 1:
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1

    train_mask[normal_train_idx] = 1
    unsup_mask[unsup_idx] = 1

    return train_mask, normal_train_idx, unsup_mask, unsup_idx
