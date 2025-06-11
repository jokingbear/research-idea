import typing

from .tuple_pipe import TuplePipe


class BaseConfigs(TuplePipe):

    def __init__(self):
        super().__init__()

        public_members = [a for a in dir(self) if a[0] != '_' and (isinstance(getattr(self, a), BaseConfigs) or not callable(getattr(self, a)))]
        self._marked_attributes.extend(public_members)

        for k, t in typing.get_type_hints(type(self)).items():
            if _is_configs(t):
                setattr(self, k, t())

    def run(self, **new_configs):
        attributes = set(self._marked_attributes)
        for update_attr, update_val in new_configs.items():
            if update_attr in attributes:
                setattr(self, update_attr, update_val)
        
            for attr in attributes:
                if attr != update_attr:
                    obj = getattr(self, attr)
                    if isinstance(obj, BaseConfigs):
                        obj.run(**new_configs)
                    elif isinstance(obj, dict):
                        if update_attr in obj:
                            obj[update_attr] = update_val
        return self

    def as_dict(self)->dict[str, typing.Any]:
        return {k:getattr(self, k) for k in self._marked_attributes}
        

def _is_configs(t: type):
    walk = True
    is_base_configs = False
    while walk:
        is_base_configs = t is BaseConfigs
        walk = not is_base_configs and t is not None
        t = t.__base__
        
    return is_base_configs
