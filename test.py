import numpy as np
import pandas as pd

shuffled_idc = np.random.choice(15, size=15, replace=False)
mapping_idc = sorted([(old, new) for new, old in enumerate(shuffled_idc)], key=lambda kv: kv[0])
inversed_idc = [new for _, new in mapping_idc]


a = np.random.randn(15)
b = a[shuffled_idc]
c = b[inversed_idc]
pd.DataFrame(np.stack([a, b, c], axis=-1), columns=["original", "shuffle", "inverse"])
