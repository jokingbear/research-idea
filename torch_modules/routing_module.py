import torch
import torch.nn as nn

rank = 2
con_op = torch.conv2d


class DynamicRouting(nn.Module):

    def __init__(self, fi, fo, n_group, n_iter=3, use_bias=True):
        super().__init__()

        self.n_group = n_group
        self.n_iter = n_iter
        self.kernels = ()
        self.bias = None

        self.build_kernels(fi, fo, n_group)
        self.build_bias(fo) if use_bias else None

    def forward(self, x):
        if self.n_iter == 1:
            ws = torch.cat(self.kernels, dim=1)
            con = con_op(x, ws, self.bias)

            return con
        else:
            g = self.n_group
            ws = torch.cat(self.kernels, dim=0)
            gf = ws.shape[0]
            con = con_op(x, ws, groups=self.n_group)
            spatial_shape = con.shape[2:]
            con = con.reshape((-1, g, gf // g, *spatial_shape))

            beta = torch.zeros((1, g, 1) + (1,) * rank)

            for i in range(self.n_iter):
                alpha = torch.sigmoid(beta)
                v = torch.sum(alpha * con, dim=(1,), keepdim=True)

                if i == self.n_iter - 1:
                    v = v[:, 0, ...]
                    return v + self.bias.reshape([1, -1] + [1]*rank) if self.bias is not None else v

                beta = beta + torch.sum(v * con, dim=(2,), keepdim=True)

    def build_kernels(self, fi, fo, n_group):
        self.kernels = []

        for i in range(n_group):
            val = torch.zeros((fo, fi) + (1,) * rank)
            val = nn.init.orthogonal_(val)
            val = nn.Parameter(val)
            self.register_parameter(f"weights_{i}", val)
            self.kernels.append(nn.Parameter(val))

    def build_bias(self, fo):
        bias = nn.Parameter(torch.zeros(fo))

        self.bias = bias
