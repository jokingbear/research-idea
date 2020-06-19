import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func


class GraphSequential(nn.Module):

    def __init__(self, node_embedding, *args):
        """
        :param node_embedding: embedding extracted from text, either numpy or torch tensor
        :param args: additional torch module for transformation
        """
        super().__init__()

        if not torch.is_tensor(node_embedding):
            node_embedding = torch.tensor(node_embedding, dtype=torch.float)

        self.embedding = nn.Parameter(node_embedding, requires_grad=False)
        self.sequential = nn.Sequential(*args)

    def forward(self):
        return self.sequential(self.embedding)


class GraphLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, correlation_matrix, bias=True):
        """
        :param in_channels: size of input features
        :param out_channels: size of output features
        :param correlation_matrix: correlation matrix for information propagation
        :param bias: whether to use bias
        """
        super().__init__(in_channels, out_channels, bias)

        assert isinstance(correlation_matrix, nn.Parameter), "correlation must be nn.Parameter"

        self.correlation_matrix = correlation_matrix

    def forward(self, x):
        prop = torch.matmul(self.correlation_matrix, x)

        return super().forward(prop)


class GCN(nn.Module):

    def __init__(self, embeddings, correlations, out_features):
        """
        :param embeddings: init embeddings for graph, either numpy or torch.tensor
        :param correlations: normalized adjacency matrix
        :param out_features: output features of extractor
        """
        super().__init__()

        self.out_features = out_features
        correlations = torch.tensor(correlations, dtype=torch.float)
        correlations = nn.Parameter(correlations, requires_grad=False)

        self.graph = GraphSequential(embeddings, *[
            GraphLinear(embeddings.shape[-1], out_features // 2, correlations),
            nn.LeakyReLU(0.2, inplace=True),
            GraphLinear(out_features // 2, out_features, correlations)
        ])

        self.bias = nn.Parameter(torch.zeros(embeddings.shape[0]), requires_grad=True)

    def forward(self, x):
        embeddings = self.graph()
        logits = func.linear(x, embeddings, self.bias)

        return logits

    def export_linear(self):
        """
        return linear layer for better test time inference
        :return: nn.Linear module
        """
        linear = nn.Linear(self.out_features, self.graph.embedding.shape[0])

        with torch.no_grad():
            linear.weight.data = self.graph()
            linear.bias.data = self.bias

        return linear


def get_label_correlation(df, columns, return_count=True):
    """
    Calculate correlation of columns from data frame
    :param df: pandas dataframe
    :param columns: colunms to calculate correlation
    :param return_count: return occurrence count
    :return: correlation and counts
    """
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
    """
    Get adjacency matrix from smoothed correlation
    :param smooth_corr: smoothed correlation matrix as dataframe
    :param neighbor_ratio: how strong neighbor nodes affect main nodes
    :return: adjacency matrix as dataframe
    """
    identiy = np.identity(smooth_corr.shape[0])
    reweight = smooth_corr - identiy
    reweight = reweight * neighbor_ratio / (1 - neighbor_ratio) / (reweight.values.sum(axis=0, keepdims=True) + 1e-8)
    reweight = reweight + identiy

    D = reweight.values.sum(axis=1) ** (-0.5)
    D = np.diag(D)
    normalized = D @ reweight.values.transpose() @ D
    return pd.DataFrame(normalized, index=smooth_corr.index, columns=smooth_corr.columns)
