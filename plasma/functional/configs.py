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
    
    def __setattr__(self, name, value):
        return super().__setattr__(name, value)


class BaseConfigs(metaclass=__meta):

    def __init__(self):
        raise NotImplementedError('this class does not support init')

    @classmethod
    def update(cls, update_dict:dict):
        for key, value in update_dict.items():
            current_value = getattr(cls, key)

            if isinstance(value, dict): 
                if isinstance(current_value, dict):
                    current_value.update(value)
                else:
                    current_value.update(value)
            else:
                setattr(cls, key, value)
        
        return cls
