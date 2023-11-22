import loralib

import plasma.search_engines as engines
import networkx as nx

db = ['tiểu đường', 'đường', 'huyết áp', 'đường huyết']

matcher = engines.SequenceMatcher(db)

query = 'tiểu đường là gì, và ảnh hưởng thế nào đến huyết áp'
matcher.match_query(query,)


import torchvision.models as pretrains
import loralib as ll
import torch.nn as nn


a = nn.Linear(18, 18, bias=True)

b = ll.Linear(18, 18, r=16)
b.load_state_dict(a.state_dict(), strict=False)