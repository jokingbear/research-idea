def run_pipe(funcs, log_fn=None, verbose=True):
    """
    run a list or dict of function sequentially, outputs of the previous function are inputed to the next function
    :param funcs: list or dict of functions to run
    :param log_fn: log the output of each function, signature (name, output)
    :param verbose: whether to print running process
    """

    if isinstance(funcs, (list, tuple)):
        funcs = {i: f for i, f in enumerate(funcs)}
    elif not isinstance(funcs, dict):
        raise NotImplemented(f'not support funcs of type {type(funcs)}')
    
    result = None
    for k in funcs:
        if verbose:
            print('running', k)

        if result is not None:
            result = funcs[k](result)
        else:
            result = funcs[k]()

        if log_fn is not None:
            log_fn(k, result)

    return result
