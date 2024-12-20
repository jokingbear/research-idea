import pandas as pd
import networkx as nx
import numpy as np

from ..functional import AutoPipe, partials, auto_map_func
from scipy.stats import hmean


class PathWalker(AutoPipe):

    def __init__(self, graph:nx.DiGraph, threshold, top_k):
        super().__init__()

        self._graph = graph
        self.threshold = threshold
        self.top_k = top_k
    
    def run(self, token_frame:pd.DataFrame):
        single_candidates = self._run_single(token_frame)
        multi_candidates = self._run_multi(token_frame)
        
        candidates = single_candidates
        if multi_candidates is not None:
            candidates = multi_candidates
            if len(single_candidates) > 0:
                candidates = pd.concat([single_candidates, multi_candidates], axis=0, ignore_index=True)
            candidates = _remove_overlap(candidates)

        candidates = candidates.groupby(['query_start_index', 'query_end_index'], as_index=False).apply(self._sort_values)
        return candidates

    def _run_single(self, token_frame:pd.DataFrame):
        data = []
        for idx, matches in token_frame[['matches']].itertuples():
            candidate = matches[matches >= self.threshold]
            for token, score in candidate.items():
                if 'candidate' in self._graph.nodes[token]:
                    data.append([(token,), idx, idx + 1, 1, score])
        
        data = pd.DataFrame(data, columns=['candidate', 'query_start_index', 'query_end_index', 'match_len', 'score'])
        return data

    def _run_multi(self, token_frame:pd.DataFrame):
        # token_frame: start_idx, end_idx, token, matches
        data = [] 
        for idx, matches in token_frame[['matches']].itertuples():
            for match_index, db_token in enumerate(matches.index):
                results = {}
                self._walk(idx, match_index, token_frame, set(), results)
                for (end_idx, end_match_idx), candidates in results.items():
                    candidate_frame = pd.DataFrame({'candidate': list(candidates)})
                    candidate_frame['query_start_index'] = idx
                    candidate_frame['query_end_index'] = end_idx + 1
                    candidate_frame['match_len'] = end_idx - idx + 1
                    candidate_frame['start_token'] = db_token
                    candidate_frame['end_token'] = token_frame.iloc[end_idx]['matches'].index[end_match_idx]
                    data.append(candidate_frame)
        
        if len(data) > 0:
            data = pd.concat(data, axis=0, ignore_index=True)
            
            aligned_frame = data[['start_token', 'end_token', 'query_start_index', 'match_len', 'candidate']]
            scorer = partials(_score_1st_match, token_frame, pre_apply_before=False)
            scorer = auto_map_func(scorer)
            scores = [scorer(row) for row in aligned_frame.itertuples(index=False)]
            data['score'] = scores
            data = data[data['score'] >= self.threshold]
            return data[['candidate', 'query_start_index', 'query_end_index', 'match_len', 'score']]

    def _walk(self, text_index:int, match_index:int, token_frame:pd.DataFrame, path_candidates:set, results: dict):
        next_index = text_index + 1
        if next_index == len(token_frame):
            if len(path_candidates) > 0:
                results[text_index, match_index] = path_candidates
        else:
            next_tokens = token_frame.iloc[next_index]['matches']
            current_token = token_frame.iloc[text_index]['matches'].index[match_index]
            for next_match_index, next_token in enumerate(next_tokens.index):
                if (current_token, next_token) in self._graph.edges:
                    full_next_candidates = self._graph.edges[current_token, next_token]['paths']

                    if len(path_candidates) > 0:
                        candidates = path_candidates.intersection(full_next_candidates)
                    else:
                        candidates = full_next_candidates

                    if len(candidates) == 0:
                        results[text_index, match_index] = path_candidates
                    else:
                        self._walk(next_index, next_match_index, token_frame, candidates, results)

    def _sort_values(self, df:pd.DataFrame):
        n = self.top_k or len(df)
        return df.sort_values(['score', 'match_len'], ascending=False).iloc[:n]


def _score_1st_match(start_token, end_token, start_query_index, match_len, candidate, token_frame):
    for i, token in enumerate(candidate):
        j = i + match_len
        if token == start_token and j <= len(candidate) and candidate[j - 1] == end_token:
            substring = candidate[i:j]
            if all(tk in token_frame.iloc[start_query_index + k]['matches'].index for k, tk in enumerate(substring)):
                scores = [token_frame.iloc[start_query_index + k]['matches'].loc[tk] for k, tk in enumerate(substring)]
                return hmean(scores)


def _remove_overlap(candidates:pd.DataFrame):
    start = candidates['query_start_index'].values
    end = candidates['query_end_index'].values

    interval_counts = (start == start[:, np.newaxis]) & (end[:, np.newaxis] == end)
    interval_counts = interval_counts.sum(axis=-1)
    start_cond = (start <= start[:, np.newaxis]) & (start[:, np.newaxis] <= end)
    end_cond = (start <= end[:, np.newaxis]) & (end[:, np.newaxis] <= end)
    cond = start_cond & end_cond
    cond = cond.sum(axis=-1) == interval_counts
    return candidates[cond]
