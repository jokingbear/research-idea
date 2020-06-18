import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GraphSequential(nn.Module):

    def __init__(self, node_embedding, *args):
        super().__init__()

        if not torch.is_tensor(node_embedding):
            node_embedding = torch.tensor(node_embedding, dtype=torch.float)

        self.embedding = nn.Parameter(node_embedding, requires_grad=False)
        self.sequential = nn.Sequential(*args)

    def forward(self):
        return self.sequential(self.embedding)


class GraphLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, correlation_matrix, bias=True):
        super().__init__(in_channels, out_channels, bias)

        self.correlation_matrix = correlation_matrix

    def forward(self, x):
        prop = torch.matmul(self.correlation_matrix, x)

        return super().forward(prop)


def get_label_correlation(df, columns, return_count=True):
    counts = pd.DataFrame(columns=columns, index=columns)

    for c1 in columns:
        for c2 in columns:
            counts.loc[c1, c2] = len(df[(df[c1] == 1) & (df[c2] == 1)])

    correlation = counts / np.diag(counts)[:, np.newaxis]

    if return_count:
        return correlation, counts
    else:
        return correlation


def get_adjacency_matrix(smooth_corr, neighbor_ratio=0.2):
    identiy = np.identity(smooth_corr.shape[0])
    reweight = smooth_corr - identiy
    reweight = reweight * neighbor_ratio / (1 - neighbor_ratio) / (reweight.values.sum(axis=0, keepdims=True) + 1e-8)
    reweight = reweight + identiy

    D = reweight.values.sum(axis=1) ** (-0.5)
    D = np.diag(D)
    normalized = D @ reweight.values.transpose() @ D
    return pd.DataFrame(normalized, index=smooth_corr.index, columns=smooth_corr.columns)
