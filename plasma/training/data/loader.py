import multiprocessing as mp
import numpy as np
import ctypes
import torch


class PrefetchLoader:

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=True, prefetch=2, pool=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.workers = prefetch
        self.pool = pool

    def __len__(self):
        iterations = len(self.dataset) // self.batch_size

        if not self.drop_last:
            iterations += min(len(self.dataset) % self.batch_size, 1)

        return iterations

    def __iter__(self):
        reset = getattr(self.dataset, "reset", None)
        if callable(reset):
            self.dataset.reset()

        iterations = len(self)

        if self.drop_last:
            indices = np.random.choice(len(self.dataset), size=iterations * self.batch_size, replace=False)
            indices = indices.reshape([-1, self.batch_size])
        else:
            indices = np.arange(0, len(self.dataset))
            indices = [indices[i * self.batch_size:(i + 1) * self.batch_size] for i in range(iterations)]

        task_q = mp.Queue()
        [task_q.put(idc) for idc in indices]

        prefetch_q = mp.Queue(maxsize=self.workers)
        processings = [mp.Value(ctypes.c_bool, True) for _ in range(self.workers)]
        processes = [mp.Process(target=self._prefetch, args=(task_q, prefetch_q, p))
                     for p in processings]
        [p.start() for p in processes]

        try:
            while True:
                if task_q.empty() & prefetch_q.empty() & (sum([p.value for p in processings]) == 0):
                    break

                results = prefetch_q.get()

                if isinstance(results, list):
                    yield [torch.tensor(r) for r in results]
                else:
                    yield torch.tensor(results)
        finally:
            [p.terminate() for p in processes]

    def _prefetch(self, task_q, prefetch_q, processing):
        pool = mp.Pool(self.pool)

        while not task_q.empty():
            processing.value = True
            indices = task_q.get()
            results = pool.map(self._read_dataset, indices)

            if isinstance(results[0], list):
                results = [np.stack([r[i] for r in results], axis=0) for i in range(len(results[0]))]
            else:
                results = np.stack(results, axis=0)

            prefetch_q.put(results)
            processing.value = False

    def _read_dataset(self, idx):
        items = self.dataset[idx]

        return self._to_numpy(items)

    def _to_numpy(self, value):
        if isinstance(value, list) or isinstance(value, tuple):
            return [self._to_numpy(v) for v in value]
        else:
            return np.array(value)
