import torch
import os

from huggingface_hub import hf_hub_download, snapshot_download
from ..meta import import_module


def download_module(repo_id: str, patterns=('*.py', '*.json', '*.yaml', '*.yml'), local_dir=None):
    revision_splits = repo_id.split('@')
    revision = None if len(revision_splits) == 1 else revision_splits[-1]

    repo_id = revision_splits[0]
    module_name = repo_id.split('-1')[-1]

    if local_dir is None:
        if 'HF_HOME' in os.environ:
            local_dir = os.environ['HF_HOME']
        else:
            local_dir = 'dependencies'

    path = f'{local_dir}/{module_name}'
    path = snapshot_download(repo_id, allow_patterns=patterns, local_dir=path, revision=revision)

    return import_module(path)


def download_checkpoint(repo_id_filename, device=None, local_dir=None):
    device = device or 'cpu'
    path = download_file(repo_id_filename, local_dir)
    return torch.load(path, map_location=device)


def download_file(repo_id_filename, local_dir=None):
    repo_id, filename_revision = repo_id_filename.split(':')

    filename_revision = filename_revision.split('@')
    filename = filename_revision[0]
    revision = None if len(filename_revision) == 1 else filename_revision[1]
    
    if local_dir is None and 'HF_HOME' in os.environ:
        local_dir = os.environ['HF_HOME']

    path = hf_hub_download(repo_id, filename, local_dir=local_dir, revision=revision)
    return path


def set_dir(path='.cache/'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    os.environ['HF_HOME'] = path
    os.environ['TRANSFORMERS_CACHE'] = path


def get_dir():
    return os.environ.get('HF_HOME', None)
