import multiprocessing as mp

from tqdm.auto import tqdm


class TqdmPool:

    def __init__(self, workers=8) -> None:
        self._process_pool = mp.Pool(workers)

    def map(self, arr, func, chunksize, **tqdm_kwargs):
        iterator = self._process_pool.imap(func, arr, chunksize)

        return [*tqdm(iterator, **tqdm_kwargs)]

    def submit_task(self, func):
        return self._process_pool.apply_async(func)
    
    def __enter__(self):
        self._process_pool.__enter__()
        return self
    
    def __exit__(self, *_):
        self._process_pool.__exit__(*_)
