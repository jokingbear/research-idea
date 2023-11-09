import pandas as pd
import numpy as np

import networkx as nx
import difflib
import re
import scipy.stats as stats

from .utils import _word_tokenize, _remove_subset, word_splitter


_sentence_splitter = re.compile('(.*?)([,;?!.]|$)')


class GraphMatcher:

    def __init__(self, db, case=False):
        if not isinstance(db, pd.Series):
            db = pd.Series(db)

        if not case:
            db = db.str.lower()

        self.db = db
        self.token_graphs = self._build_graph()

        self.case = case

    def match_query(self, query: str, k=None, threshold=0.85, marginal_threshold=0.7):
        assert marginal_threshold >= 0.5, (f'the minimum matching should be bigger than 0.5, '
                                           f'currently {marginal_threshold}')

        if not self.case:
            query = query.lower()

        sentence_groups = [g for g in _sentence_splitter.finditer(query) if len({*g.span(0)}) > 1]
        total_candidates = []
        for sentence_group in sentence_groups:
            start, end = sentence_group.span(0)

            sentence_tokens = _word_tokenize(sentence_group.group(0))
            candidates_mappings = self._find_token_candidates(sentence_tokens, marginal_threshold)
            candidates_subgraph = self._build_subgraph(candidates_mappings)
            candidates = _list_all_candidates(candidates_mappings, candidates_subgraph)
            candidates = self._cleanup(candidates, threshold, sentence_tokens)

            candidates['start_idx'] += start
            candidates['end_idx'] += start
            total_candidates.append(candidates)

        total_candidates = pd.concat(total_candidates, axis=0, ignore_index=True)
        total_candidates = total_candidates.sort_values('score', ascending=False)

        if k is not None:
            total_candidates = total_candidates.iloc[:k]

        return total_candidates

    def _build_graph(self):
        token_paths = self.db.map(word_splitter.findall)

        graph = nx.DiGraph()
        for path in token_paths:
            if len(path) == 1:
                graph.add_node(path[0])
            else:
                for i in range(1, len(path)):
                    graph.add_edge(path[i - 1], path[i])

        return graph

    def _find_token_candidates(self, sentence_tokens, marginal_threshold):
        mappings = []
        for stk in sentence_tokens['token']:
            scores = pd.Series({etk: difflib.SequenceMatcher(None, stk, etk).ratio() for etk
                                in self.token_graphs.nodes})
            scores = scores.sort_values(ascending=False)
            scores = scores[scores >= marginal_threshold]

            mappings.append({
                'sentence_token': stk,
                'entity_tokens': scores.index.values,
                'scores': scores.values,
            })

        return pd.DataFrame(mappings)

    def _build_subgraph(self, mappings):
        entity_tokens = np.concatenate(mappings['entity_tokens'], axis=0)
        entity_tokens = np.unique(entity_tokens)
        return self.token_graphs.subgraph(entity_tokens)

    def _cleanup(self, candidates, threshold, sentence_tokens):
        if threshold is not None:
            candidates = candidates[candidates['score'] >= threshold]

        candidates = candidates[candidates['entity'].isin(self.db)]
        candidates = _remove_subset(candidates)
        candidates['start_idx'] = sentence_tokens.iloc[candidates['start_idx']]['start_idx'].values
        candidates['end_idx'] = sentence_tokens.iloc[candidates['end_idx'] - 1]['end_idx'].values

        return candidates


def _list_all_candidates(mappings, subgraph):
    total_paths = []
    for i, row in mappings.iterrows():
        paths = []
        for token, score in zip(row['entity_tokens'], row['scores']):
            paths.append([(token,), score])
            _walk_path(token, mappings['entity_tokens'].values[i + 1:], mappings['scores'].values[i + 1:], subgraph,
                       [token], [score], paths)
        paths = pd.DataFrame(paths, columns=['path', 'scores'])
        paths['start_idx'] = i
        paths['end_idx'] = i + paths['path'].map(len)
        paths['entity'] = paths['path'].map(' '.join)
        total_paths.append(paths)

    total_paths = pd.concat(total_paths, axis=0, ignore_index=True)
    total_paths['score'] = total_paths['scores'].map(stats.hmean)

    return total_paths[['entity', 'score', 'start_idx', 'end_idx']]


def _walk_path(start, next_token_sequences, next_score_sequences, graph: nx.DiGraph, path, scores, token_acc):
    if len(next_token_sequences) == 0:
        return
    else:
        neighbors = set(graph.neighbors(start))
        for new_start, new_score in zip(next_token_sequences[0], next_score_sequences[0]):
            if new_start in neighbors:
                new_path = (*path, new_start)
                new_scores = (*scores, new_score)
                token_acc.append([new_path, new_scores])
                _walk_path(new_start, next_token_sequences[1:], next_score_sequences[1:], graph,
                           new_path, new_scores, token_acc)
