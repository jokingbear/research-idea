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
