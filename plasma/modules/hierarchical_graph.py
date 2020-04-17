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


class GraphAdaBatchNorm(nn.Module):

    def __init__(self, n_node, n_embedding, batchnorm_module):
        super().__init__()

        for p in batchnorm_module.parameters():
            p.requires_grad = False

        self.transformer = nn.Linear(n_node * n_embedding, 2 * batchnorm_module.num_features, bias=False)
        self.normalization = batchnorm_module

        self.gamma_bias = nn.Parameter(batchnorm_module.weight.data, requires_grad=False)
        self.beta_bias = nn.Parameter(batchnorm_module.bias.data, requires_grad=False)
        batchnorm_module.reset_parameters()

    def forward(self, x, embeddings):
        rank = len(x.shape[2:])
        normalized = self.normalization(x)

        gamma_beta = self.transformer(embeddings.view(1, -1)).view(2, -1)
        gamma = (gamma_beta[0] + self.gamma_bias).view(1, -1, *[1]*rank)
        beta = (gamma_beta[1] + self.beta_bias).view(1, -1, *[1]*rank)

        return gamma * normalized + beta
