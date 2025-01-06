import networkx as nx
import pandas as pd
import difflib

from ..functional import AutoPipe


class TokenMatcher(AutoPipe):

    def __init__(self, graph: nx.DiGraph, threshold):
        super().__init__()

        self._graph = graph
        self.threshold = threshold
    
    def run(self, tokens:list[str]):
        matches = []
        for tk in tokens:
            scores = {db_tk: difflib.SequenceMatcher(None, tk, db_tk).ratio() for db_tk in self._graph.nodes}
            scores = pd.Series(scores)
            matches.append(scores[scores >= self.threshold])

        return matches
