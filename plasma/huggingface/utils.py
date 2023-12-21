import torch

from huggingface_hub import hf_hub_download, snapshot_download
from ..meta import import_module


def download_module(repo_id, patterns=('*.py', '*.json', '*.yaml'), local_dir='dependencies'):
    module_name = repo_id.split('/')[-1]

    path = f'{local_dir}/{module_name}'
    path = snapshot_download(repo_id, allow_patterns=patterns, local_dir=path)

    return import_module(path)


def download_checkpoint(repo_id_filename, local_dir=None):
    repo_id, filename = repo_id_filename.split(':')
    path = hf_hub_download(repo_id, filename, local_dir=local_dir)

    return torch.load(path, map_location='cpu')
