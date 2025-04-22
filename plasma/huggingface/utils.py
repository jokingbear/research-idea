import torch
import os
import re

from huggingface_hub import hf_hub_download, snapshot_download
from ..meta import import_module


def download_module(repo_id: str, patterns=('*.py', '*.json', '*.yaml', '*.yml'), local_dir=None):
    revision_splits = repo_id.split('@')
    revision = None if len(revision_splits) == 1 else revision_splits[-1]

    repo_id = revision_splits[0]
    module_name = repo_id.split('-1')[-1]

    if local_dir is None:
        local_dir = os.environ.get('HF_HOME', 'dependencies')

    path = f'{local_dir}/{module_name}'
    path = snapshot_download(repo_id, allow_patterns=patterns, local_dir=path, revision=revision)

    return import_module(path)


def download_checkpoint(repo_id_filename, device=None, local_dir=None):
    device = device or 'cpu'
    path = download_file(repo_id_filename, local_dir)
    return torch.load(path, map_location=device, weights_only=True)


def download_file(repo_id_filename, local_dir=None):
    repo_id, filename_revision = repo_id_filename.split(':')

    filename_revision = filename_revision.split('@')
    filename = filename_revision[0]
    revision = None if len(filename_revision) == 1 else filename_revision[1]
    
    if local_dir is None:
        local_dir = os.environ.get('HF_HOME', None)

    path = hf_hub_download(repo_id, filename, local_dir=local_dir, revision=revision)
    return path


def set_dir(path='.cache/'):
    if not os.path.exists(path):
        os.makedirs(path)
    
    os.environ['HF_HOME'] = path
    os.environ['TRANSFORMERS_CACHE'] = path


def get_dir():
    return os.environ.get('HF_HOME', None)


def download_folder(repo_id_folder, local_dir=None, repo_type='dataset'):
    matched = re.search(r'(.*?):([^@]+)(@.+){0,1}', repo_id_folder)
    repo_id = matched.group(1)
    folder_name = matched.group(2)
    revision = matched.group(3)

    if local_dir is None:
        local_dir = os.environ.get('HF_HOME', '.cache/')
    local_dir = f'{local_dir}/{repo_id}'
    
    path = snapshot_download(repo_id, revision=revision, repo_type=repo_type, 
                             allow_patterns=[folder_name], local_dir=local_dir)
    path = path + '/' + re.sub(r'\*.*', '', folder_name)
    return path
