import plasma.parallel_processing as pp
import time
import multiprocessing as mp

from tqdm.auto import tqdm


def haha(i, q):
    x = q.get()
    time.sleep(0.5)
    print(x)


if __name__ == '__main__':
    with pp.ThreadCommunicator([haha] * 1) as pcomm:
        for i in tqdm(range(100)):
            pcomm.queue.put_nowait(i)
            time.sleep(0.1)
