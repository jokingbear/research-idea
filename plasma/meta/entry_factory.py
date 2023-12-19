_factory_ = {}


def entry_point(func_or_class):
    _factory_[func_or_class.__module__] = func_or_class
    return func_or_class


def get_module_entry(module):
    return _factory_.get(module, None)
