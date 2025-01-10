def test_append(x:list):
    x.append(5)
    return x

def test_exception(x:list):
    raise KeyError('dafad')
