from turtle import forward
import torch
import torch.nn as nn


class AlgebraicModule(nn.Module):

    def __add__(self, another):
        assert isinstance(another, nn.Module), 'Algebraic module does not support (+) operation with other type beside nn.Module'
        return OpModule(self, another, '+')
    
    def __sub__(self, another):
        assert isinstance(another, nn.Module), 'Algebraic module does not support (-) operation with other type beside nn.Module'
        return OpModule(self, another, '-')
    
    def __mul__(self, another):
        assert isinstance(another, nn.Module), 'Algebraic module does not support (*) operation with other type beside nn.Module'
        return OpModule(self, another, '*')
    
    def __truediv__(self, another):
        assert isinstance(another, nn.Module), 'Algebraic module does not support (/) operation with other type beside nn.Module'
        return OpModule(self, another, '/')


class OpModule(AlgebraicModule):

    def __init__(self, module1, module2, op) -> None:
        super().__init__()

        assert op in {'+', '-', '*', '/'}, 'only support +, -, *, / ops'
        self.module1 = module1
        self.module2 = module2
        self.op = op
    
    def forward(self, *args, **kwargs):
        result1 = self.module1(*args, **kwargs)
        result2 = self.module2(*args, **kwargs)

        if self.op == '+':
            return result1 + result2
        elif self.op == '*':
            return result1 * result2
        elif self.op == '/':
            return result1 / result2
        elif self.op == '-':
            return result1 - result2

    def extra_repr(self) -> str:
        return f'op={self.op},' + super().extra_repr()