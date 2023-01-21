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
    
    funcs = {k: auto_func(funcs[k]) for k in funcs}

    result = None
    for k in funcs:
        if verbose:
            print('running', k)

        result = funcs[k](result)

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
