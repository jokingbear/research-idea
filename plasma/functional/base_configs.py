from .auto_pipe import AutoPipe


class BaseConfigs(AutoPipe):

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

