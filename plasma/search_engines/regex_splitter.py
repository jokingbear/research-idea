import re
import pandas as pd

from ..functional import AutoPipe


class RegexTokenizer(AutoPipe):

    def __init__(self, pattern):
        super().__init__()

        self.pattern = re.compile(pattern)

    def run(self, string: str):
        matches = [{'start_idx': m.span(0)[0], 'end_idx': m.span(0)[1], 'token': m.group(0)}
                   for m in self.pattern.finditer(string)]

        return pd.DataFrame(matches)
