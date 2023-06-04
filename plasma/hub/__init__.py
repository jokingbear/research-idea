from .hub_entries import HubEntries
from pathlib import Path
from ..functional import run_pipe
from functools import partial


def import_module(path):
    """
    get enty point of a hub folder
    :param path: path to python module
    :return: HubEntries
    """
    path = Path(path)
    return HubEntries(path.parent, path.name.replace(".py", ""))


def run_module(path, name, *args, **kwargs):
    entries = import_module(path)
    return entries.load(name, *args, **kwargs)
