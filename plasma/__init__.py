from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

notebook = False


def get_tqdm(*args, **kwargs):
    pbar = tqdm_nb if notebook else tqdm
    return pbar(*args, **kwargs)
