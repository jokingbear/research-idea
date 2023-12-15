from .module_hub import ModuleHub
from pathlib import Path
from .entry_factory import entry_point, get_module_entry


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
    if 'path' not in cfg:
        raise AttributeError('there is no path in config file')

    path = cfg['path']
    name = cfg.get('name', None)

    inspector = import_module(path)
    kwargs = {k: cfg[k] for k in cfg if k not in ['name', 'path']}

    if name is not None:
        return inspector.load(name, **kwargs)
    elif get_module_entry(inspector.module.__name__) is not None:
        entry = get_module_entry(inspector.module.__name__)
        return entry(**kwargs)
    else:
        raise AttributeError(f'Cant find entry point of module {path}, either add an entry decorator or specified'
                             f'entry method/class in cfg name')
