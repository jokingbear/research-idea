import torch
import torch.nn as nn


class DynamicRouting(nn.Module):

    def __init__(self, in_filters, out_filters, groups, iters=3, rank=2, bias=True):
        super().__init__()

        spatial_shape = [1] * rank
        shape = [out_filters, in_filters * groups, *spatial_shape]
        self.groups = groups
        self.iters = iters
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(*shape))
        self.bias = nn.Parameter(torch.Tensor(1, out_filters, *spatial_shape)) if bias else None
        self.con_op = torch.conv2d if rank == 2 else torch.conv3d if rank == 3 else torch.convolution

        self.reset_parameters()

    def forward(self, x):
        fo, gfi = self.weight.shape[:2]
        g = self.groups
        fi = gfi // g
        rank = self.rank
        con_op = self.con_op

        if self.iters == 1:
            con = con_op(x, self.weight, self.bias)

            return con
        else:
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

                beta = beta + torch.sum(v * con, dim=(2,), keepdim=True)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def extra_repr(self):
        fo = self.weight.shape[0]
        fi = self.weight.shape[1] // self.groups

        return f"in_filters={fi}, out_filters={fo}, groups={self.groups}, iters={self.iters}, rank={self.rank}" \
               f", bias={self.bias is not None}"
