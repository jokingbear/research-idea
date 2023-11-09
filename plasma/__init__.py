import plasma.utils
from .utils import get_tqdm


def set_tqdm(notebook=True):
    utils.notebook = notebook
