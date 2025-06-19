import plasma.functional as F
import pandas as pd
import networkx as nx
import numpy as np

from .regex_splitter import RegexTokenizer
from .token_matcher import TokenMatcher
from .path_set_walker2 import PathWalker


class GraphIndexer(F.SimplePipe[str, pd.DataFrame]):
    
    def __init__(self, data:list[str], 
                 group_splitter=r'([^\.\n]+)', tokenizer=r'(\w+)', 
                 token_threshold=0.7, top_k=5):
        super().__init__()
        
        assert len(data) == len(set(data)), 'data is not unique'
        
        self.sentence_splitter = RegexTokenizer(group_splitter)
        self.token_splitter = RegexTokenizer(tokenizer)
        
        graph, tokenized_data = self._build_graph(data)
        self._data = tokenized_data
        self.token_matcher = TokenMatcher(graph, token_threshold)
        self.path_walker = PathWalker(graph, top_k)
    
    def _build_graph(self, data):
        graph = nx.DiGraph()
        
        dbs = []
        for i, txt in enumerate(data):
            token_frame = self.token_splitter.run(txt.lower())
            path = tuple(token_frame['token'].tolist())
            nx.add_path(graph, path)
            
            for tk in token_frame['token']:
                paths:set = graph.nodes[tk].get('paths', set())
                paths.add(path)
                graph.add_node(tk, paths=paths)

            dbs.append([i, txt, path])
        
        return graph, pd.DataFrame(dbs, columns=['data_index', 'text', 'path'])

    def run(self, inputs):
        sentence_frame = self.sentence_splitter.run(inputs.lower()).rename(columns={'token': 'sentence'})
        path_frames = [self._run_sentence(s) for s in sentence_frame['sentence']]
        path_frames = [f for f in path_frames if len(f) > 0]
        
        columns = ['query_start_idx', 'query_end_idx', 'data_index', 'original', 'substring_matching_score', 'coverage_score', 'matched_len']
        if len(path_frames) > 0:
            path_frames = pd.concat(path_frames, axis=0, ignore_index=True)
            path_frames = path_frames.merge(sentence_frame, on='sentence')
            path_frames['query_start_idx'] = path_frames['start_idx'] + path_frames['query_start_idx']
            path_frames['query_end_idx'] = path_frames['start_idx'] + path_frames['query_end_idx']
        else:
            path_frames = pd.DataFrame(columns=columns)
            
        path_frames = path_frames[columns]\
            .groupby(['query_start_idx', 'query_end_idx'])\
                .apply(_post_process, self.path_walker.top_k, include_groups=False)
        
        if len(path_frames) > 0:
            path_frames = path_frames.droplevel(None)
            path_frames = _find_largest_intervals(path_frames)

        return path_frames
    
    def _run_sentence(self, sentence:str):
        token_frame = self.token_splitter.run(sentence)
        token_scores = self.token_matcher.run(token_frame['token'])
        token_frame['matches'] = token_scores
        path_frame = self.path_walker.run(token_frame)
        path_frame['sentence'] = sentence
        path_frame = path_frame.rename(columns={'matched_score': 'substring_matching_score'})
        path_frame = path_frame.merge(self._data, left_on='candidate', right_on='path')
        return path_frame.rename(columns={'matched_score': 'substring_matching_score', 'text': 'original'})


def _post_process(df:pd.DataFrame, topk):
    return df.sort_values(['substring_matching_score', 'coverage_score'], ascending=False).iloc[:topk]


def _find_largest_intervals(df:pd.DataFrame):
    intervals = {(s, e) for s, e, *_ in df.index}
    intervals = np.array(list(intervals))
    
    cond_counts = ((intervals[:, 0] <= intervals[:, 0, np.newaxis]) & (intervals[:, 1, np.newaxis] <= intervals[:, 1])).sum(axis=1)
    cond_counts = cond_counts == 1
    filtered = intervals[cond_counts]
    return df.loc[[(int(s), int(e)) for s, e in filtered]].sort_index()
