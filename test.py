import numpy as np

a = np.arange(0, 10)

shuffle_idc = np.random.choice(len(a), size=len(a), replace=False)

b = a[shuffle_idc]

tmp = [(e, i) for i, e in enumerate(shuffle_idc)]
inverse_shuffle_idc = [i for _, i in sorted(tmp, key=lambda kv: kv[0])]
