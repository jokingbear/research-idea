import torch
import torch.nn.functional as func

from plasma.modules.configs import *


class DynamicRouting(nn.Module):

    def __init__(self, in_filters, out_filters, groups, iters=3, bias=True):
        super().__init__()

        spatial_shape = [1] * rank
        shape = [out_filters, in_filters * groups, *spatial_shape]
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.groups = groups
        self.iters = iters
        self.weight = nn.Parameter(torch.zeros(*shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_filters), requires_grad=True) if bias else None

        self.reset_parameters()

    def forward(self, x):
        if self.iters == 1:
            con = con_op(x, self.weight, self.bias)

            return con
        else:
            fo = self.out_filters
            fi = self.in_filters
            g = self.groups

            weight = self.weight.reshape([fo, g, fi] + [1] * rank).transpose(0, 1).reshape([g * fo, fi] + [1] * rank)
            con = con_op(x, weight, groups=self.groups)
            spatial_shape = con.shape[2:]
            con = con.reshape([-1, g, fo, *spatial_shape])

            beta = torch.zeros([], device=x.device)

            for i in range(self.iters):
                alpha = torch.sigmoid(beta)

                v = torch.sum(alpha * con, dim=(1,), keepdim=True)

                if i == self.iters - 1:
                    v = v[:, 0, ...]
                    return v if self.bias is None else v + self.bias

                v = func.normalize(v, dim=2)
                beta = beta + torch.sum(v * con, dim=(2,), keepdim=True)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def extra_repr(self):
        fo = self.out_filters
        fi = self.in_filters

        return f"in_filters={fi}, out_filters={fo}, groups={self.groups}, iters={self.iters}, " \
               f"bias={self.bias is not None}"


class EMRouting(nn.Module):

    def __init__(self, in_channels, out_channels, clusters=3, groups=32, iters=3, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.clusters = clusters
        self.groups = groups
        self.iters = iters

        self.weight = nn.Parameter(torch.zeros(clusters * out_channels, in_channels * groups, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(clusters * out_channels), requires_grad=bias)

        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        if self.iters == 1:
            return con_op(x, self.weight, self.bias)
        else:
            weight = self.weight.view(-1, self.groups, self.in_channels, 1, 1)
            weight = weight.transpose(0, 1).view(-1, self.in_channels, 1, 1)  # GCO, I

            # B, GCO, H, W
            con = con_op(x, weight, groups=self.groups)

            # B, G, C, O, H, W
            con = con.view(-1, self.groups, self.clusters, self.out_channels, *x.shape[2:])
            coeff = torch.ones(1, self.groups, self.clusters, 1, *con.shape[2:], device=con.device) / self.groups

            for i in range(self.iters):
                mean = (coeff * con).sum(dim=1, keepdim=True)
                var = (coeff * con ** 2).sum(dim=1, keepdim=True) - mean ** 2 + 1e-7

                if i == self.iters - 1:
                    return mean[:, 0, 0, ...]

                pi = coeff.sum(dim=1, keepdim=True) / self.groups
                log_p = -((con - mean).pow(2) / var + var.log()) / 2
                coeff = (log_p * pi.log()).softmax(dim=2)
