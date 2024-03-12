import plasma.parallel_processing as pp
import torch
import time

from tqdm.auto import tqdm


def haha(i, q):
    x = q.get()
    a = torch.tensor([1] * x, device='cuda:0')
    print(a.sum())
    del a
    q.task_done()


if __name__ == '__main__':
    with pp.TorchCommunicator(haha, start_method='spawn', auto_loop=True) as comm:
        for i in tqdm(range(100)):
            comm.queue.put_nowait(i)
            time.sleep(0.1)

        comm.queue.join()
    print('done all task')
