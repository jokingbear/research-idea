from .class_registrator import ObjectFactory

_factory = ObjectFactory()


def entry_point(func_or_class):
    return _factory.register(func_or_class.__module__)(func_or_class)


def get_module_entry(module):
    return _factory[module]
