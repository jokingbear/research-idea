import networkx as nx
import pandas as pd
import scipy.stats as stats

from ..functional import AutoPipe


class PathStack(AutoPipe):

    def __init__(self, index, curr_token, curr_score, prev_path):
        super().__init__()

        self.index = index
        self.token = curr_token
        self.prev_path = prev_path
        length = 1
        inverse_score = 1 / curr_score
        if prev_path is not None:
            length += len(prev_path)
            inverse_score += len(prev_path) / prev_path.score
        inverse_score = inverse_score / length

        self.score = 1 / inverse_score
        self.length = length

    def __len__(self):
        return self.length

    def to_list(self, out=[]):
        if self.prev_path is not None:
            self.prev_path.to_list(out)
        out.append(self.token)
        return out


class PathWalker(AutoPipe):

    def __init__(self, threshold, graph:nx.DiGraph, data:pd.DataFrame) -> None:
        super().__init__()
        self.threshold = threshold
        self._graph = graph
        self._data = set(data['standardized_text_path'])

    def run(self, sequential_db_tokens:list[pd.Series]):
        paths = self._walk_path(sequential_db_tokens)
        paths = [(tuple(p.to_list([])), p.score, p.index) for p in paths]
        paths = pd.DataFrame(paths, columns=['path', 'score', 'token_index'])
        return paths

    def _walk_path(self, sequential_db_tokens:list[pd.Series]) -> list[PathStack]:
        running_paths = {}
        finished_paths = []

        for i, tokens in enumerate(sequential_db_tokens):
            new_paths = {}
            for token, score in tokens.items():
                new_root = PathStack(i, token, score, None)
                new_paths[token] = [new_root]
                if (token,) in self._data:
                    finished_paths.append(new_root)

                for prev_token in running_paths:
                    if (prev_token, token) in self._graph.edges:
                        for prev_path in running_paths[prev_token]:
                            new_path = PathStack(prev_path.index, token, score, prev_path)
                            new_paths[token].append(new_path)
                            if new_path.score >= self.threshold:
                                finished_paths.append(new_path)
            running_paths = new_paths
        return finished_paths
