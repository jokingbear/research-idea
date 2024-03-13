import time

from queue import Queue


def sync_queues_determine(q1: Queue, q2: Queue, total, return_items=False):
    items = []
    for _ in range(total):
        x = q1.get()
        items.append(x)
        q2.put(x)
    
    if return_items:
        return items


def sync_queues_undetermine(q1: Queue, q2: Queue, delay_time=0.5, loop_time=0.1, return_items=False):
    if delay_time is not None:
        time.sleep(delay_time)
    
    counter = 0
    items = []
    while q1.qsize() > 0:
        x = q1.get()
        q2.put(x)
        if loop_time is not None:
            time.sleep(loop_time)
        counter += 1
        items.append(x)
    
    if return_items:
        return counter, items

    return counter
