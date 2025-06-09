import pandas as pd
import networkx as nx
import numpy as np

from ..functional import AutoPipe, partials, auto_map_func
from scipy.stats import hmean
from ..networkx import directed_intersect, Merger
from itertools import product


class PathWalker(AutoPipe):

    def __init__(self, graph:nx.DiGraph, top_k):
        super().__init__()

        self._graph = graph
        self.top_k = top_k
    
    def run(self, token_frame:pd.DataFrame):
        candidate_graph = _construct_graph(token_frame)
        intersected_graph = directed_intersect(self._graph, candidate_graph, 
                                               scores=Merger.SECOND, paths=Merger.FIRST)
        position_graph = construct_position_graph(intersected_graph)

        candidates = [_resolve_components(position_graph, c) for c in nx.connected_components(position_graph.to_undirected())]
        candidates = sum(candidates, [])
        candidates = pd.DataFrame(candidates, columns=['start', 'end', 'candidate', 'matched_score', 'matched_len',
                                                       'coverage_score'])\
                        .groupby(['start', 'end'])\
                            .apply(lambda tdf: tdf.sort_values(['matched_score', 'coverage_score'], ascending=False).iloc[:self.top_k], 
                                   include_groups=False)
        candidates['query_start_idx'] = token_frame.iloc[candidates.index.get_level_values(0).values]['start_idx'].values
        candidates['query_end_idx'] = token_frame.iloc[candidates.index.get_level_values(1).values - 1]['end_idx'].values
        candidates = candidates.reset_index(drop=True)
        return candidates
    

def _construct_graph(token_frame:pd.DataFrame):
    graph = nx.DiGraph()
    for i, matches in enumerate(token_frame['matches']):
        next_matches = pd.Series()
        if i + 1 < len(token_frame):
            next_matches = token_frame.iloc[i + 1]['matches']
        
        for m, score in matches.items():
            if m in graph:
                scores = graph.nodes[m].get('scores', {})
            else:
                scores = {}
            scores[i] = score
            graph.add_node(m, scores=scores)
            
            for nm in next_matches.index:
                graph.add_edge(m, nm)
    return graph


def construct_position_graph(g:nx.DiGraph):
    pg = nx.DiGraph()
    for n in g:
        scores = pd.Series(g.nodes[n]['scores'])
        paths = g.nodes[n]['paths']
        for i, s in scores.items():
            pg.add_node((i, n), score=s, paths=paths)
    
    for n, m in g.edges:
        n_scores = pd.Series(g.nodes[n]['scores'])
        m_scores = pd.Series(g.nodes[m]['scores'])
        
        filtered_scores = n_scores[(n_scores.index + 1).isin(m_scores.index)]
        for i, s in filtered_scores.items():
            pg.add_edge((i, n), (i + 1, m))
    return pg


def _resolve_components(g:nx.DiGraph, component_nodes):
    if len(component_nodes) == 1:
        n, = [*component_nodes]
        total_candidates = []
        score = g.nodes[n]['score']
        for p in g.nodes[n]['paths']:
            total_candidates.append([n[0], n[0] + 1, p, score, 1, 1 / len(p)])
    else:
        h:nx.DiGraph = g.subgraph(component_nodes)
        roots = [n for n in h if h.in_degree(n) == 0]
        leaves = [n for n in h if h.out_degree(n) == 0]
        
        total_candidates = []
        for r, l in product(roots, leaves):
            for p in nx.all_simple_paths(h, r, l):
                if p[-1][0] - p[0][0] + 1 == len(p):
                    score = hmean([h.nodes[n]['score'] for n in p])
                    candidates = set(h.nodes[p[0]]['paths']).intersection(*[h.nodes[n]['paths'] for n in p])
                    for c in candidates:
                        matched_len = l[0] - r[0] + 1
                        total_candidates.append([r[0], l[0] + 1, c, score, matched_len, matched_len / len(c)])
        
    return total_candidates
