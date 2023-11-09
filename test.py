import plasma.search_engines as engines
import networkx as nx

db = ['tiểu đường', 'đường', 'huyết áp', 'đường huyết']

matcher = engines.SequenceMatcher(db)

query = 'tiểu đường là gì, và ảnh hưởng thế nào đến huyết áp'
matcher.match_query(query,)
