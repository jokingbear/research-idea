import os
import yaml
import json

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb


notebook = False


def get_tqdm(iterator=None, total=None, desc=None, show=True):
    pbar = tqdm_nb if notebook else tqdm
    return pbar(iterator, total=total, desc=desc, disable=not show)


def set_tqdm(is_notebook=True):
    global notebook
    notebook = is_notebook


def set_os_environment(config_file):
    with open(config_file, 'rb') as handle:
        if '.yaml' in config_file:
            cfg = yaml.safe_load(handle)
        elif '.json' in config_file:
            cfg = json.load(handle)
        else:
            raise NotImplementedError('only support .json and .yaml file at the moment')

    for k, v in cfg.items():
        os.environ[k] = v
