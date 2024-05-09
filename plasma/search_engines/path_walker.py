import networkx as nx
import pandas as pd
import scipy.stats as stats

from ..functional import AutoPipe


class PathWalker(AutoPipe):

    def __init__(self, threshold, graph:nx.DiGraph, data:pd.DataFrame) -> None:
        super().__init__()
        self.threshold = threshold
        self._graph = graph
        self._data = data

    def run(self, sequential_db_tokens:list[pd.Series]):
        paths = self._walk_path(sequential_db_tokens)
        
        path_df = []
        for (reversed_index, _), local_paths in paths.items():
            local_paths = pd.DataFrame(local_paths, columns=['path', 'score'])
            local_paths = local_paths[local_paths['score'] >= self.threshold]
            local_paths['token_index'] = len(sequential_db_tokens) - reversed_index - 1
            path_df.append(local_paths)
        
        if len(path_df) == 0:
            return pd.DataFrame(columns=['path', 'score', 'token_index'])

        path_df = pd.concat(path_df, axis=0, ignore_index=True)
        path_df['path'] = path_df['path'].map(_render_path)
        return path_df

    def _walk_path(self, sequential_db_tokens:list[pd.Series]):
        paths = {}
        reversed_sequential_db_tokens = sequential_db_tokens[::-1]
        for i in range(len(reversed_sequential_db_tokens)):
            mapped_tokens = reversed_sequential_db_tokens[i]
            for tk, score in mapped_tokens.items():
                local_paths = [([tk], score)]
            
                if i > 0:
                    next_tokens = reversed_sequential_db_tokens[i - 1].keys()
                    for ntk in next_tokens:
                        if (tk, ntk) in self._graph.edges:
                            for next_path, next_score in paths[i - 1, ntk]:
                                new_score = (i + 1) / (i / next_score + 1 / score)
                                local_paths.append(([tk, next_path], new_score))

                paths[i, tk] = local_paths
        return paths


def _render_path(path):
    if len(path) == 1:
        return path[0],
    
    start, nexts = path

    return (start, *_render_path(nexts))
