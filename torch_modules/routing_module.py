import torch
import torch.nn as nn

rank = 2
con_op = torch.conv2d


class DynamicRouting(nn.Module):

    def __init__(self, fi, fo, n_group, n_iter=3, bias=True):
        super().__init__()

        shape = [fo * n_group, fi] + [1] * rank
        self.n_group = n_group
        self.n_iter = n_iter
        self.weight = nn.Parameter(torch.Tensor(*shape))
        self.bias = nn.Parameter(torch.Tensor(fo)) if bias else None

        self.reset_parameters()

    def forward(self, x):
        if self.n_iter == 1:
            gfo, fi = self.weight.shape[:2]
            g = self.n_group
            fo = gfo // g
            w = self.weight.reshape([g, fo, fi] + [1] * rank).transpose(0, 1).reshape([fo, -1] + [1] * rank)
            con = con_op(x, w, self.bias)

            return con
        else:
            g = self.n_group
            gf = self.weight.shape[0]
            con = con_op(x, self.weight, groups=self.n_group)
            spatial_shape = con.shape[2:]
            con = con.reshape((-1, g, gf // g, *spatial_shape))

            beta = torch.zeros([], device=x.device)

            for i in range(self.n_iter):
                alpha = torch.sigmoid(beta)
                v = torch.sum(alpha * con, dim=(1,), keepdim=True)

                if i == self.n_iter - 1:
                    v = v[:, 0, ...]
                    return v + self.bias.reshape([1, -1] + [1]*rank) if self.bias is not None else v

                beta = beta + torch.sum(v * con, dim=(2,), keepdim=True)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def extra_repr(self):
        fo = self.weight.shape[0] // self.n_group
        fi = self.weight.shape[1]

        return f"fi={fi}, fo={fo}, n_group={self.n_group}, n_iter={self.n_iter}, bias={self.bias is not None}"
