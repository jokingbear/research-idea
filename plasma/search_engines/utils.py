import pandas as pd

import re


word_splitter = re.compile(r'(\w+)')


def _word_tokenize(string: str):
    tokens = [{'start_idx': word_group.span(0)[0],
               'end_idx': word_group.span(0)[1],
               'token': word_group.group(0)} for word_group in word_splitter.finditer(string)]

    return pd.DataFrame(tokens)


def _remove_subset(matches: pd.DataFrame):
    matches = matches.groupby(['start_idx', 'end_idx'], group_keys=False,
                              as_index=False).apply(lambda df: df.iloc[df['score'].argmax()])
    keeps = []
    for idx, row in matches.iterrows():
        start_idx = row['start_idx']
        end_idx = row['end_idx']
        cond = (matches['start_idx'] <= start_idx) & (end_idx <= matches['end_idx'])
        if len(matches[cond]) < 2:
            keeps.append(idx)

    return matches.loc[keeps]
