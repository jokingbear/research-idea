import warnings


class ObjectFactory(dict):

    def register(self, name, class_or_func):
        if name in self:
            warnings.warn(f'many entry points for: {name}, overriding the current one')
        self[name] = class_or_func

        return class_or_func
