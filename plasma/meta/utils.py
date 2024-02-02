import importlib

from .module_hub import ModuleHub
from pathlib import Path
from .entry_factory import get_module_entry


def import_module(path, verbose=True):
    """
    get enty point of a hub folder
    :param path: path to python module
    :param verbose: be verbose in importing module
    :return: HubEntries
    """
    path = Path(path)
    module = ModuleHub(path.parent, path.name.replace(".py", ""))

    if verbose:
        print(f'finished importing {path}')

    return module


def run_cfg(cfg: dict):
    """
    run a module based on config
    Args:
        cfg: config dict, must have keys: path, name
    Returns:
        loaded function or class
    """
    kwargs = {k: cfg[k] for k in cfg if k not in ['name', 'path']}
    entry = load_entry(cfg)
    return entry(**kwargs)


def load_entry(cfg: dict):
    """
    load an entry from a module
    Args:
        cfg: config dict, must have key path
    Returns:

    """
    if 'path' not in cfg:
        raise AttributeError('there is no path in config file')

    path = cfg['path']
    name = cfg.get('name', None)

    inspector = import_module(path)

    if isinstance(name, str):
        return inspector[name]
    elif get_module_entry(inspector.module.__name__) is not None:
        return get_module_entry(inspector.module.__name__)
    else:
        raise AttributeError(f'Cant find entry point of module {path}, either add an entry decorator or specified'
                             f'entry method/class in cfg name')


def load_entries(**cfg):
    """
    load keys of entries
    Args:
        **cfg: dict of entries from different modules
    Returns:
        dict
    """
    loaded = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = load_entry(v)
        loaded[k] = v

    return loaded


def mass_import(file, pattern):
    path = Path(file)
    name = path.name.replace('.py', '')
    if name == '__init__':
        name = path.parent.name

    for p in path.parent.glob(pattern):
        reader_name = p.name.replace('.py', '')
        importlib.import_module(f'.{reader_name}', name)
