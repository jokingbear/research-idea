from .hub_entries import HubEntries
from pathlib import Path
from ..functional import run_pipe
from functools import partial


def get_entries(path):
    """
    get enty point of a hub folder
    :param path: path to python module
    :return: HubEntries
    """
    path = Path(path)
    return HubEntries(path.parent, path.name.replace(".py", ""))


def run_modules(cfgs):
    modules = [import_module(cfg) for cfg in cfgs]

    return run_pipe(modules)


def import_module(cfg):
    path = cfg['path']
    name = cfg['name']

    kwargs = {k: cfg[k] for k in cfg if k not in {'path', 'name'}}

    entries = get_entries(path)
    entry = partial(entries.load, name)
    return entry, kwargs
