import time

from queue import Queue


def transit_queue(total, input_queue: Queue, output_queue: Queue):
    counters = 0
    while counters != total:
        x = input_queue.get()
        output_queue.put(x)


def transfer_queue(q1: Queue, q2: Queue, delay_time=0.5, loop_time=0.1):
    if delay_time is not None:
        time.sleep(delay_time)
    
    counter = 0
    while q1.qsize() > 0:
        x = q1.get()
        q2.put(x)
        if loop_time is not None:
            time.sleep(loop_time)
        counter += 1
    
    return counter
