import re
import pandas as pd

from ..functional import AutoPipe


class RegexTokenizer(AutoPipe):

    def __init__(self, pattern):
        super().__init__()

        self.pattern = re.compile(pattern)

    def run(self, string: str):
        matches = [(*m.span(0), m.group(0)) for m in self.pattern.finditer(string)]
        
        return pd.DataFrame(matches, columns=['start_idx', 'end_idx', 'token'])
