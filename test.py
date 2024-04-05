import re

from pathlib import Path

packages = [re.sub(r'\\|\/', '.', str(p.parent)) for p in Path('plasma').rglob('__init__.py')]


