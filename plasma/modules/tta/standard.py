import torch
import torch.nn as nn


class Flip(nn.Module):

    def __init__(self, dims):
        """
        Create a Flip tta module
        Args:
            dims: flip dims
        """
        super().__init__()

        self.dims = dims

    def forward(self, x, reverse=False):
        return x.flip(dims=self.dims)

    def extra_repr(self):
        return f'dim={self.dims}'


class Compose(nn.Module):

    def __init__(self, main_module, aug_modules):
        """
        :param main_module: main computation module
        :param aug_modules: test time augmentation modules
        """
        super().__init__()

        assert len(aug_modules) > 0, "must have at least 1 augmentation module"

        self.main_module = main_module
        self.aug_modules = nn.ModuleList(aug_modules)

    def forward(self, x):
        augs = torch.cat([x] + [aug_module(x, reverse=False) for aug_module in self.aug_modules], dim=0)
        results = self.main_module(augs)
        results = results.view(1 + len(self.aug_modules), *x.shape)

        if results.shape[1:] == x.shape:
            results = [results[0]] + [aug_module(r, reverse=True)
                                      for aug_module, r in zip(self.aug_modules, results[1:])]
            results = torch.stack(results, dim=0)

        return results
