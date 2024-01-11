import warnings
_factory_ = {}


def entry_point(func_or_class):
    if func_or_class.__module__ in _factory_:
        warnings.warn(f'many entry points of module: {func_or_class.__module__}, '
                      f'ModuleHub will use the last registered one.')
    _factory_[func_or_class.__module__] = func_or_class
    return func_or_class


def get_module_entry(module):
    return _factory_.get(module, None)
