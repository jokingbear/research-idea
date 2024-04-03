import plasma.torch as ptorch


class A(ptorch.modules.AlgebraicModule):
    pass


a = A()
b = A()

print(a + b)