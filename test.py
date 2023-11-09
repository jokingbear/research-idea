import plasma.search_engines as engines
import networkx as nx

db = ['tiểu đường', 'đường', 'huyết áp', 'đường huyết']

graph_matcher = engines.GraphMatcher(db, case=False)

query = 'tiểu đường là gì, và ảnh hưởng thế nào đến huyết áp'
graph_matcher.match_query(query)
