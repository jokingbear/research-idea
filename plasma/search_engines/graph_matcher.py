import pandas as pd
import numpy as np

import networkx as nx
import difflib
import scipy.stats as stats

from ..functional import AutoPipe
from .regex_splitter import RegexTokenizer


class GraphMatcher(AutoPipe):

    def __init__(self, texts, group_splitter=RegexTokenizer('([^:\n,;?!.]+)'), tokenizer=RegexTokenizer(r'(\w+)'),
                 token_threshold=0.5, path_threshold=0.8,
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

        standardized_texts = texts.str.lower()
        token_paths = []
        for txt in standardized_texts:
            path = tokenizer.run(txt)['token'].tolist()
            token_paths.append(tuple(path))
        graph = nx.DiGraph()
        [nx.add_path(graph, p) for p in token_paths]

        self._graph = graph
        self._data = pd.DataFrame({
            'original_text': texts.values,
            'standardized_text_path': token_paths
        })

        self.tokenizer = tokenizer
        self.group_splitter = group_splitter
        self.token_threshold = token_threshold
        self.path_threshold = path_threshold
        self.select_largest_interval = select_largest_interval
        self.top_k = top_k

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
            return pd.DataFrame([])

        total_candidates = pd.concat(total_candidates, axis=0, ignore_index=True)

        if len(total_candidates) == 0:
            return pd.DataFrame([])

        total_candidates['matched_len'] = total_candidates['query_end_index'] - total_candidates['query_start_index']
        total_candidates = (total_candidates.groupby(['query_start_index', 'query_end_index'])
                            .apply(lambda df: df.sort_values(['substring_matching_score', 'word_coverage_score'],
                                                             ascending=False).iloc[:self.top_k or len(df)]))
        total_candidates = total_candidates.drop(columns=['query_start_index', 'query_end_index'])
        if len(total_candidates.index.names) > 2:
            total_candidates = total_candidates.droplevel(2)

        if self.select_largest_interval:
            total_candidates = self._remove_subset(total_candidates)
        return total_candidates

    def _analyze_group(self, group: str):
        group_tokens = self.tokenizer.run(group)
        candidate_db_mappings = self._compare_tokens(group_tokens['token'])
        candidate_paths = self._analyze_path(candidate_db_mappings)
        
        if len(candidate_paths) == 0:
            return pd.DataFrame([])
        
        mapped_candidates = self._map_data(candidate_paths)
        candidates = self._standardize_data(mapped_candidates, group_tokens)
        return candidates

    def _compare_tokens(self, tokens):
        candidate_tokens = []
        for t in tokens:
            scores = {db_token: difflib.SequenceMatcher(None, a=t, b=db_token).ratio()
                      for db_token in self._graph.nodes}
            scores = pd.Series(scores)
            scores = scores[scores >= self.token_threshold]

            candidate_tokens.append({
                'token': t,
                'db_tokens': scores.sort_values(ascending=False),
            })

        return pd.DataFrame(candidate_tokens)

    def _analyze_path(self, mappings):
        total_db_tokens = np.concatenate(mappings['db_tokens'].map(lambda tokens: tokens.index))
        subgraph = self._graph.subgraph(total_db_tokens)

        candidates = []
        for i, row in mappings.iterrows():
            start_candidates = []
            for db_token, score in row['db_tokens'].items():
                if score >= self.path_threshold:
                    start_candidates.append(([db_token], score))
                sub_sequence_steps = mappings['db_tokens'].iloc[i + 1:].tolist()
                self._walk_path(db_token, sub_sequence_steps, [db_token], [score],
                                subgraph, start_candidates)
            candidates += [{'token_index': i, 'token': row['token'],
                            'path': tuple(p), 'score': s} for p, s in start_candidates]

        return pd.DataFrame(candidates)

    def _walk_path(self, node, next_sequences, current_path, current_scores, graph: nx.DiGraph, accumulators):
        if len(next_sequences) > 0:
            for next_node, score in next_sequences[0].items():
                if graph.has_edge(node, next_node):
                    new_path = [*current_path, next_node]
                    new_scores = [*current_scores, score]
                    total_score = stats.hmean(new_scores)

                    if total_score >= self.path_threshold:
                        accumulators.append((new_path, total_score))

                    self._walk_path(next_node, next_sequences[1:], new_path, new_scores,
                                    graph, accumulators)

    def _map_data(self, candidate_paths):
        remaining_paths = candidate_paths
        data_text = self._data['standardized_text_path'].map(' '.join)
        remaining_candidates = []
        for _, row in remaining_paths.iterrows():
            text = ' '.join(row['path'])

            temp_candidates = self._data[data_text.str.contains(rf'(\s|^){text}(\s|$)', regex=True)].reset_index()
            temp_candidates['token_index'] = row['token_index']
            temp_candidates['token'] = row['token']
            temp_candidates['score'] = row['score']
            temp_candidates['path'] = [row['path']] * len(temp_candidates)
            remaining_candidates.append(temp_candidates)

        candidates = pd.concat(remaining_candidates, axis=0, ignore_index=True)
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
            token_paths.append({'query_start_index': start_index, 'query_end_index': end_index,
                                'word_coverage_score': coverage_score})

        token_paths = pd.DataFrame(token_paths)
        final_data = pd.concat([candidates, token_paths], axis=1)
        final_data = final_data[['query_start_index', 'query_end_index', 'data_index', 'original_text',
                                 'substring_matching_score', 'word_coverage_score']]
        return final_data

    def _remove_subset(self, candidates):
        keeps = []
        start_indices = candidates.index.get_level_values('query_start_index')
        end_indices = candidates.index.get_level_values('query_end_index')

        for i, (start, end) in enumerate(candidates.index):
            is_contained = (start_indices <= start) & (end < end_indices)
            is_contained |= (start_indices < start) & (end <= end_indices)
            is_contained = np.any(is_contained)

            if not is_contained:
                keeps.append(i)

        return candidates.iloc[keeps]
