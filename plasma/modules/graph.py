import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
import pydot

from .commons import Identity


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

    def __init__(self, backbone, embeddings, correlations, backbone_features, ratio=0.5, sigmoid=True):
        """
        :param embeddings: init embeddings for graph, either numpy or torch.tensor
        :param correlations: normalized adjacency matrix in numpy
        :param backbone_features: output features of extractor
        """
        super().__init__()

        self.backbone = backbone

        correlations = torch.tensor(correlations, dtype=torch.float)
        correlations = nn.Parameter(correlations, requires_grad=False)
        bottleneck = int(np.round(backbone_features * ratio))
        self.graph = GraphSequential(embeddings, *[
            GraphLinear(embeddings.shape[-1], bottleneck, correlations),
            nn.LeakyReLU(0.2, inplace=True),
            GraphLinear(bottleneck, backbone_features, correlations),
        ])

        self.out = nn.Sigmoid() if sigmoid else Identity()

        self.bias = nn.Parameter(torch.zeros(embeddings.shape[0]), requires_grad=True)
        self.backbone_features = backbone_features

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.graph()
        logits = func.linear(features, embeddings, self.bias)

        result = self.out(logits)

        return result

    def export_linear(self):
        """
        return new gcn with graph replaced with a linear layer
        :return: nn.Sequential module
        """
        linear = nn.Linear(self.backbone_features, self.graph.embedding.shape[0])
        graph = self.graph.eval()

        with torch.no_grad():
            linear.weight.data = graph()
            linear.bias.data = self.bias.data

        model = nn.Sequential(*[
            self.backbone,
            linear,
            self.out
        ])

        return model


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
    identity = np.identity(smooth_corr.shape[0])
    reweight = smooth_corr - identity
    reweight = reweight * neighbor_ratio / (1 - neighbor_ratio) / (reweight.values.sum(axis=0, keepdims=True) + 1e-8)
    reweight = reweight + identity

    D = reweight.values.sum(axis=1) ** (-0.5)
    D = np.diag(D)
    normalized = D @ reweight.values.transpose() @ D
    return pd.DataFrame(normalized, index=smooth_corr.index, columns=smooth_corr.columns)


def get_graph(corr, threshold=0.4):
    """
    draw a pydot graph of correlation
    :param corr: dataframe of correlation matrix
    :param threshold: threshold to prune correlation
    :return: pydot graph
    """
    smooth_corr = corr >= threshold
    graph = pydot.Dot(graph_type='digraph')

    for c1 in corr.columns:
        node1 = pydot.Node(c1)
        graph.add_node(node1)

        for c2 in corr.columns:
            if c2 != c1:
                node2 = pydot.Node(c2)

                if smooth_corr.loc[c1, c2] != 0:
                    edge = pydot.Edge(node1, node2, label=np.round(corr.loc[c1, c2], decimals=2))
                    graph.add_edge(edge)

    return graph
