import test1
import time

from plasma.training import data
from torch.utils.data import DataLoader

ds = test1.Check()

loader = data.PrefetchLoader(ds, 32, drop_last=False)

tmp = []
start = time.time()
for x in loader:
    time.sleep(1)
    tmp.append(x)
end = time.time()
print(end - start)
print([t.shape for t in tmp])
