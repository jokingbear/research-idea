def as_tuple(cls):
    
    def __iter__(self):
        class_dict:dict = type(self).__dict__
        instance_dict:dict = self.__dict__
        
        final_dict = class_dict.copy()
        final_dict.update(instance_dict)
        for k, v in final_dict.items():
            if k[0] != '_':
                yield v
            
    cls.__iter__ = __iter__
    return cls
