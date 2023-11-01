def auto_map_func(func):

    def auto_input(inputs):
        if isinstance(inputs, (tuple, list)):
            return func(*inputs)
        elif isinstance(inputs, dict):
            return func(**inputs)
        elif inputs is None:
            return func()
        else:
            return func(inputs)
    
    return auto_input
