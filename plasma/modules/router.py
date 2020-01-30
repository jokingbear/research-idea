import torch
import torch.nn as nn
import torch.nn.functional as func


def dynamic_routing(x, groups, iters, bias):
    channels = x.shape[1] // groups
    spatial = x.shape[2:]
    rank = len(spatial)
    x = x.view(-1, groups, channels, *spatial)
    beta = 0

    for i in range(iters):
        alpha = torch.sigmoid(beta)
        v = (alpha * x).sum(dim=1, keepdim=True)

        if i == iters - 1:
            v = v[:, 0, ...]
            return v if bias is None else v + bias.view(-1, channels, *([1] * rank))

        v = func.normalize(v, dim=2)
        beta = beta + torch.sum(v * x, dim=2, keepdim=True)


def em_routing(x, clusters, groups, iters, bias=None, epsilon=1e-7):
    batch = x.shape[0]
    spatial = x.shape[2:]
    rank = len(spatial)
    x = x.view(batch, groups, clusters, -1, *spatial)

    r_ik = torch.ones(1, *x.shape[1:]) / clusters

    for i in range(iters):
        r_k = r_ik.sum(dim=1, keepdim=True) + epsilon
        mean = (r_ik * x).sum(dim=1, keepdim=True) / r_k

        if i == iters - 1:
            mean = mean.view(batch, -1, *spatial)
            return mean if bias is None else mean + bias.view(1, -1, *[1] * rank)

        var = (r_ik * x.pow(2)).sum(dim=1, keepdim=True) / r_k - mean.pow(2) + epsilon
        pi_k = r_k / groups
        log_p = -((x - mean).pow(2) / var + var.log()) / 2
        r_ik = (log_p + pi_k.log()).softmax(dim=2)


class DynamicRouting2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=32, iters=3, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.iters = iters

        self.weight = nn.Parameter(torch.zeros(out_channels, groups * in_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        if self.iters == 1:
            con = torch.conv2d(x, self.weight, self.bias)
            return con
        else:
            weight = self.weight.view(-1, self.groups, self.in_channels, 1, 1)
            weight = weight.transpose(0, 1).view(-1, self.in_channels, 1, 1)
            con = torch.conv2d(x, weight, groups=self.groups)

            return dynamic_routing(con, self.groups, self.iters, self.bias)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, groups={self.groups}, " \
               f"iters={self.iters}, bias={self.bias is not None}"


class EMRouting2d(nn.Module):

    def __init__(self, in_channels, out_channels, clusters=3, groups=32, iters=3, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.clusters = clusters
        self.groups = groups
        self.iters = iters

        self.weight = nn.Parameter(torch.zeros(clusters * out_channels, groups * in_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(clusters * out_channels), requires_grad=True) if bias else None

        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        if self.iters == 1:
            con = torch.conv2d(x, self.weight, self.bias)
            return con
        else:
            weight = self.weight.view(-1, self.groups, self.in_channels, 1, 1)
            weight = weight.transpose(0, 1).view(-1, self.in_channels, 1, 1)
            con = torch.conv2d(x, weight, groups=self.groups)

            return em_routing(con, self.clusters, self.groups, self.iters, self.bias)

# TODO: check implementations
