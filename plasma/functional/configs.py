class __meta(type):
    
    def __repr__(cls):
        formatter = ' ' * 2
        members = [cls.__name__]
        sub_configs = []
        for member in dir(cls):
            value = getattr(cls, member)
            if member[0] != '_' and value.__class__.__name__ != 'method':
                if value.__class__.__qualname__ == '__meta':
                    reps = repr(value).split('\n')
                    reps = [formatter + r for r in reps]
                    reps[0] = '\n' + reps[0]
                    sub_configs.extend(reps)
                else:
                    members.append(f'{formatter}{member} = {value}')
        return '\n'.join(members + sub_configs)


class BaseConfigs(metaclass=__meta):

    def __init__(self):
        raise NotImplementedError('this class does not support init')

    @classmethod
    def update(cls, **update_keys):
        for member in dir(cls):
            value = getattr(cls, member)
            
            if member[0] != '_' and value.__class__.__name__ != 'method':
                if value.__class__.__qualname__ == '__meta':
                    value.update(**update_keys)
                elif isinstance(value, dict):
                    value.update({k:v for k, v in update_keys.items() if k in value})
                elif member in update_keys:
                    setattr(cls, member, update_keys[member])
        
        return cls
