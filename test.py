import plasma.parallel_processing as pp
import time

arr = [0.5] * 100

pp.parallel_iterate(arr, time.sleep, batchsize=2)
