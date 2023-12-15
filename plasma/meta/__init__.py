from .module_entry import ModuleEntry
from pathlib import Path


def get_entries(path):
    """
    get enty point of a hub folder
    :param path: path to python module
    :return: HubEntries
    """
    path = Path(path)
    return ModuleEntry(path.parent, path.name.replace(".py", ""))


def run_cfg(cfg):
    if 'path' not in cfg:
        raise AttributeError('there is no path in config file')

    if 'name' not in cfg:
        raise AttributeError('there is no name in config file')

    path = cfg['path']
    name = cfg['name']

    entries = get_entries(path)
    kwargs = {k: cfg[k] for k in cfg if k not in ['name', 'path']}
    return entries.load(name, **kwargs)
