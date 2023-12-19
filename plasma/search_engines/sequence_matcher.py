import pandas as pd
import difflib

from ..utils import get_tqdm
from ..functional import partials
from .utils import _word_tokenize, _remove_subset


class SequenceMatcher:

    def __init__(self, db, trunks=None, case=False):
        if not isinstance(db, pd.Series):
            db = pd.Series(db)

        if not case:
            db = db.str.lower()

        self.db = db

        if isinstance(trunks, list):
            trunks = ''.join(trunks)

        self.trunks = trunks
        self.case = case

    def match_query(self, query: str, k=None, threshold=0.85, pbar=True, normalize_by_word=True):
        matches = self.search_db(query, normalize_by_word, pbar)

        if len(matches) == 0:
            return matches

        if threshold is not None:
            matches = matches[matches['score'] >= threshold]

        matches = matches.sort_values('score', ascending=False)
        matches = _remove_subset(matches)

        if k is not None:
            matches = matches.iloc[:k]

        return matches

    def search_db(self, query, normalize_by_word, pbar):
        if not self.case:
            query = query.lower()

        tokens = _word_tokenize(query) if normalize_by_word else None

        matches = []
        for entity in get_tqdm(self.db, show=pbar):
            matches.append(self._compare_string(query, entity, tokens))

        return pd.DataFrame(matches)

    def _compare_string(self, query, entity, tokens):
        if self.trunks is not None:
            check_trunk = partials(_check_trunk, self.trunks)
        else:
            check_trunk = None

        s = difflib.SequenceMatcher(check_trunk, a=query, b=entity)
        blocks = s.get_matching_blocks()
        blocks = [b for b in blocks if b.size > 0]

        matching_results = {'entity': entity}
        if len(blocks) > 0:
            start_idx = blocks[0].a
            end_idx = blocks[-1].a + blocks[-1].size

            if tokens is not None:
                lower_cond = (tokens['start_idx'] <= start_idx) & (start_idx <= tokens['end_idx'])
                start_idx = tokens[lower_cond].iloc[0]['start_idx']

                upper_cond = (tokens['start_idx'] <= end_idx) & (end_idx <= tokens['end_idx'])
                end_idx = tokens[upper_cond].iloc[0]['end_idx']

            filtered_query = query[start_idx:end_idx]
            new_matcher = difflib.SequenceMatcher(a=filtered_query, b=entity)
            matching_results['score'] = new_matcher.ratio()
            matching_results['start_idx'] = start_idx
            matching_results['end_idx'] = end_idx

        return matching_results


def _check_trunk(x, trunks):
    return x in trunks
