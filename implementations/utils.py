def standardize_kernel_stride_dilation(dimensions, check_type, value):
    if type(value) is int:
        return (value,) * dimensions
    elif (type(value) is list or type(value) is tuple) and len(value) == dimensions:
        return tuple(value)
    else:
        raise TypeError(f"{value} is not a valid {check_type}")

