import torch

from huggingface_hub import hf_hub_download, snapshot_download
from ..meta import import_module


def download_module(repo_id, patterns=('*.py', '*.json', '*.yaml', '*.yml'), local_dir='dependencies'):
    repo_id, module_name_revision = repo_id.split('/')[-1]

    module_name_revision = module_name_revision.split('@')
    module_name = module_name_revision[0]
    revision = None if len(module_name_revision) == 1 else module_name_revision[1]

    path = f'{local_dir}/{module_name}'
    path = snapshot_download(f'{repo_id}/{module_name}', allow_patterns=patterns,
                             local_dir=path, revision=revision)

    return import_module(path)


def download_checkpoint(repo_id_filename, device='cpu', local_dir=None):
    path = download_file(repo_id_filename, local_dir)
    return torch.load(path, map_location=device)


def download_file(repo_id_filename, local_dir=None):
    repo_id, filename_revision = repo_id_filename.split(':')

    filename_revision = filename_revision.split('@')
    filename = filename_revision[0]
    revision = None if len(filename_revision) == 1 else filename_revision[1]

    path = hf_hub_download(repo_id, filename, local_dir=local_dir, revision=revision)
    return path
