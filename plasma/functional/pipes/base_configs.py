from .pipe import AutoPipe


class BaseConfigs(AutoPipe):

    def __init__(self):
        super().__init__()

        public_members = [a for a in dir(self) if a[0] != '_' and (isinstance(getattr(self, a), BaseConfigs) or not callable(getattr(self, a)))]
        self._marked_attributes.extend(public_members)

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

    def to_dict(self):
        results = {}
        for attr in self._marked_attributes:
            obj = getattr(self, attr)

            if isinstance(obj, BaseConfigs):
                obj = obj.to_dict()
            results[attr] = obj

        return results
