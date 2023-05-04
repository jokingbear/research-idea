def run_pipe(funcs, log_fn=None, verbose=True):
    """
    Run a list or dict of function sequentially, outputs of the previous function are 
    inputed to the next function

    Args:
        funcs (list or dict): list or dict of functions to run sequentially
        log_fn (function, optional): logging function to be run after each step. Signature (step name, step results) -> (). 
        Defaults to None.
        verbose (bool, optional): whether to print out which step is being run. Defaults to True.

    Raises:
        NotImplemented: raise when funcs is not the type of list, tuple or dict

    Returns:
        Any: result of the final step
    """    

    if isinstance(funcs, (list, tuple)):
        funcs = {i: f for i, f in enumerate(funcs)}
    elif not isinstance(funcs, dict):
        raise NotImplemented(f'not support funcs of type {type(funcs)}')
    
    funcs = {k: standardize_func_inputs(funcs[k]) for k in funcs}

    result = None
    for k in funcs:
        if verbose:
            print('running', k)

        func, args, kwargs = funcs[k]

        if result is None:
            result = func(*args, **kwargs)
        else:
            result = func(result, *args, **kwargs)

        if log_fn is not None:
            log_fn(k, result)

    return result


def auto_func(func):

    def auto_input(inputs):
        if isinstance(inputs, tuple):
            return func(*inputs)
        elif isinstance(inputs, dict):
            return func(**inputs)
        elif inputs is None:
            return func()
        else:
            return func(inputs)
    
    return auto_input


def standardize_func_inputs(func_inputs):
    func = None
    args = []
    kwargs = {}

    if isinstance(func_inputs, (list, tuple)):
        if len(func_inputs) == 1:
            func, = func_inputs
        elif len(func_inputs) == 2:
            func, tmp = func_inputs

            if isinstance(tmp, dict):
                kwargs = tmp
            else:
                args = tmp
        elif len(func_inputs) == 3:
            func, args, kwargs = func_inputs
        else:
            raise NotImplementedError(f'Unsupported input type {[type(i) for i in func_inputs]}')
    else:
        func = func_inputs
    
    return func, args, kwargs
