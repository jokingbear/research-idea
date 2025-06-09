import networkx as nx

from enum import Flag, auto


class Merger(Flag):
    FIRST = auto()
    SECOND = auto()
    

def directed_intersect(g1:nx.Graph, g2:nx.Graph, **keep_attr:Merger) -> nx.DiGraph:
    g:nx.Graph = nx.intersection(g1, g2)
    
    removals = []
    for e in g.edges:
        if e not in g1.edges or e not in g2.edges:
            removals.append(e)
    g.remove_edges_from(removals)
    
    for n in g:
        for k, v in keep_attr.items():
            
            if Merger.FIRST in v:
                g.add_node(n, **{k: g1.nodes[n][k]})
            
            if Merger.SECOND in v:
                if k in g.nodes[n]:
                    g.nodes[n][k] = [g.nodes[n][k], g2.nodes[n][k]]
                else:
                    g.add_node(n, **{k:g2.nodes[n][k]})
    
    return g.to_directed()
