import pandas as pd
import numpy as np

import networkx as nx
import difflib
import scipy.stats as stats

from ..functional import AutoPipe, partials
from .regex_splitter import RegexTokenizer
from ..logging import Timer
from concurrent.futures import ProcessPoolExecutor
from ..parallel_processing import TqdmPool
from .path_walker_2 import PathWalker
from tqdm.contrib.concurrent import thread_map


class GraphMatcher(AutoPipe):

    def __init__(self, texts, group_splitter=r'([^:\n,;?!.]+)', tokenizer=r'(\w+)',
                 token_threshold=0.6, path_threshold=0.8,
                 select_largest_interval=True, top_k=None):
        """
        Args:
            texts: list of texts
            group_splitter: regex for extract group from query text
            token_threshold: threshold for comparing token
            path_threshold: threshold for comparing matched path
            select_largest_interval: whether the engine should remove results contained in other results
            top_k: filter top k for each match
        """
        super().__init__()
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)

        if isinstance(tokenizer, str):
            tokenizer = RegexTokenizer(tokenizer)
        
        if isinstance(group_splitter, str):
            group_splitter = RegexTokenizer(group_splitter)

        self.tokenizer = tokenizer
        self.group_splitter = group_splitter 
        self.token_threshold = token_threshold
        self.select_largest_interval = select_largest_interval
        self.top_k = top_k
        self._graph, self._data = self._build_graph(texts)
        self.path_walker = PathWalker(path_threshold, self._graph, self._data)

    def _build_graph(self, texts:pd.Series):
        standardized_texts = texts.str.lower()
        token_paths = []
        for txt in standardized_texts:
            path = self.tokenizer.run(txt)['token'].tolist()
            token_paths.append(tuple(path))
        graph = nx.DiGraph()
        [nx.add_path(graph, p) for p in token_paths]

        data = pd.DataFrame({
            'original_text': texts.values,
            'standardized_text_path': token_paths
        })
        return graph, data

    def run(self, query: str):
        standardized_query = query.lower()
        groups = self.group_splitter.run(standardized_query)
        total_candidates = []
        for _, g in groups.iterrows():
            start = g['start_idx']
            candidates = self._analyze_group(g['token'])
            if len(candidates) > 0:
                candidates['query_start_index'] += start
                candidates['query_end_index'] += start
                total_candidates.append(candidates)

        if len(total_candidates) == 0:
            total_candidates = pd.DataFrame([], columns=[
                'query_start_index', 'query_end_index', 'substring_matching_score', 'word_coverage_score'
            ])
        else:
            total_candidates = pd.concat(total_candidates, axis=0, ignore_index=True)
        
        total_candidates['matched_len'] = total_candidates['query_end_index'] - total_candidates['query_start_index']
        total_candidates = total_candidates.groupby(['query_start_index', 'query_end_index'])
        total_candidates = total_candidates.apply(self._sort_frame, include_groups=False)

        if len(total_candidates.index.names) > 2:
            total_candidates = total_candidates.droplevel(2)

        if self.select_largest_interval:
            total_candidates = self._remove_subset(total_candidates)
        
        return total_candidates

    def _analyze_group(self, group: str):
        group_tokens = self.tokenizer.run(group)
        candidate_db_mappings = self._compare_tokens(group_tokens['token'])
        candidate_paths = self._analyze_path(candidate_db_mappings)
        mapped_candidates = self._map_data(candidate_paths)
        
        candidates = []
        if mapped_candidates is not None:
            candidates = self._standardize_data(mapped_candidates, group_tokens)
        
        return candidates

    def _compare_tokens(self, tokens):
        candidate_tokens = []
        for t in tokens:
            scores = {db_token: difflib.SequenceMatcher(None, a=t, b=db_token).ratio()
                      for db_token in self._graph.nodes}
            scores = pd.Series(scores)
            scores = scores[scores >= self.token_threshold]

            candidate_tokens.append((t, scores.sort_values(ascending=False)))

        return pd.DataFrame(candidate_tokens, columns=['token', 'db_tokens'])

    def _analyze_path(self, mappings):
        candidates = self.path_walker.run(mappings['db_tokens'].tolist())
        candidates['token'] = [mappings.iloc[i]['token'] for i in candidates['token_index']] 

        return candidates

    def _map_data(self, candidate_paths):
        data_text = self._data['standardized_text_path'].map(' '.join)
        data_text = ' ' + data_text + ' '

        candidates = [_map_contain(index_row, data_text, self._data) for index_row in candidate_paths.iterrows()]
        if len(candidates) > 0:
            candidates = pd.concat(candidates, axis=0, ignore_index=True)
            candidates = candidates.rename(columns={'score': 'substring_matching_score', 'index': 'data_index'})

            return candidates

    def _standardize_data(self, candidates, tokens):
        token_paths = []
        for _, row in candidates.iterrows():
            path_len = len(row['path'])
            standardized_text_path_len = len(row['standardized_text_path'])
            coverage_score = path_len / standardized_text_path_len

            index = row['token_index']
            start_index = tokens.iloc[index]['start_idx']
            end_index = tokens.iloc[index + path_len - 1]['end_idx']
            token_paths.append((start_index, end_index, coverage_score))

        token_paths = pd.DataFrame(token_paths, columns=['query_start_index', 'query_end_index', 'word_coverage_score'])
        final_data = pd.concat([candidates, token_paths], axis=1)
        final_data = final_data[['query_start_index', 'query_end_index', 'data_index', 'original_text',
                                 'substring_matching_score', 'word_coverage_score']]
        return final_data

    def _sort_frame(self, candidates:pd.DataFrame):
        candidates = candidates.sort_values(['substring_matching_score', 'word_coverage_score'], ascending=False)

        if self.top_k is not None:
            candidates = candidates.iloc[:self.top_k]
        return candidates

    def _remove_subset(self, candidates):
        start_indices = candidates.index.get_level_values('query_start_index').values
        end_indices = candidates.index.get_level_values('query_end_index').values

        bound_below = (start_indices <= start_indices[:, np.newaxis]) & (end_indices[:, np.newaxis] < end_indices)
        bound_above = (start_indices < start_indices[:, np.newaxis]) & (end_indices[:, np.newaxis] <= end_indices)
        is_contained = bound_below | bound_above
        is_contained = np.any(is_contained, axis=-1)

        return candidates[~is_contained]


def _map_contain(index_row, data_text, data):
    row = index_row[1]
    text = ' '.join(row['path'])
    temp_candidates = data[data_text.str.contains(f' {text} ', regex=False)].reset_index()
    temp_candidates['token_index'] = row['token_index']
    temp_candidates['token'] = row['token']
    temp_candidates['score'] = row['score']
    temp_candidates['path'] = [row['path']] * len(temp_candidates)
    return temp_candidates
