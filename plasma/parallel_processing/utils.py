def transit_queue(total, input_queue, output_queue):
    counters = 0
    while counters != total:
        x = input_queue.get()
        output_queue.put(x)
