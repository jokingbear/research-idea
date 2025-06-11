import networkx as nx
import pandas as pd

from ..functional import AutoPipe
from .regex_splitter import RegexTokenizer
from .token_matcher import TokenMatcher
from .path_set_walker import PathWalker
from warnings import deprecated



@deprecated('this class is deprecated, please use graph indexer')
class GraphMatcher(AutoPipe):

    def __init__(self, data:list[str], group_splitter=r'([^\.\n]+)', 
                 tokenizer=r'(\w+)', token_threshold=0.7, path_threshold=0.8, top_k=5):
        super().__init__()
        assert len(data) == len(set(data)), 'data is not unique'
        
        self.sentence_splitter = RegexTokenizer(group_splitter)
        self.token_splitter = RegexTokenizer(tokenizer)

        tokenized_data, graph = self._build_graph(data)
        self._data = tokenized_data
        self.token_matcher = TokenMatcher(graph, token_threshold)
        self.walker = PathWalker(graph, path_threshold, top_k)
    
    def _build_graph(self, data):
        graph = nx.DiGraph()

        tokenized_data = []
        for i, item in enumerate(data):
            original = item
            item = item.lower()
            tokens = self.token_splitter.run(item)
            path = tuple(tokens['token'].tolist())
            
            assert len(path) > 0, f'cannot tokenize {item} at index {i}'
            tokenized_data.append([i, original, path])

            if len(path) > 1:
                nx.add_path(graph, path)
                for i, node in enumerate(path):
                    if i + 1 < len(path):
                        next_node = path[i + 1]
                        created_paths = graph.edges[node, next_node].get('paths', set())
                        created_paths.add(path)
                        graph.edges[node, next_node]['paths'] = created_paths
            else:
                graph.add_node(path[0], candidate=path)

        return pd.DataFrame(tokenized_data, columns=['data_index', 'original', 'tokenized']), graph

    def run(self, query:str):
        query = query.lower()
        str_groups = self.sentence_splitter.run(query)
        candidates = []
        for start_idx, _, text in str_groups.itertuples(index=False):
            group_candidates = self._run_text(start_idx, text)
            if len(group_candidates) > 0:
                candidates.append(group_candidates)

        if len(candidates) > 0:
            candidates = pd.concat(candidates, axis=0, ignore_index=True)
        else:
            colums = ['query_start_index', 'query_end_index', 'data_index', 'original', 'substring_matching_score', 'matched_len', 'coverage_score']
            candidates = pd.DataFrame([], columns=colums)
       
        sorting_criteria = ['substring_matching_score', 'coverage_score']
        candidates = candidates.groupby(['query_start_index', 'query_end_index']).apply(lambda df: df.sort_values(sorting_criteria, ascending=False),
                                                                                        include_groups=False)
        try:
            candidates = candidates.droplevel(level=None)
        finally:
            return candidates
    
    def _run_text(self, start_idx:int, text:str):
        token_frame = self.token_splitter.run(text)
        token_frame['matches'] = self.token_matcher.run(token_frame['token'])
        candidates = self.walker.run(token_frame)
        
        candidates['query_start_index'] = token_frame.iloc[candidates['query_start_index']]['start_idx'].values + start_idx
        candidates['query_end_index'] = token_frame.iloc[candidates['query_end_index'] - 1]['end_idx'].values + start_idx
        candidates = candidates.merge(self._data, left_on='candidate', right_on='tokenized')

        colums = ['query_start_index', 'query_end_index', 'data_index', 'original', 'score', 'matched_len', 'coverage_score']
        candidates = candidates[colums].rename(columns={'score': 'substring_matching_score'})
        return candidates
