import re

from setuptools import setup
from pathlib import Path

packages = [re.sub(r'\\|\/', '.', str(p.parent)) for p in Path('plasma').rglob('__init__.py')]

setup(
    name='plasma',
    version='2.0.11c11',
    packages=['plasma', *packages],
    url='https://github.com/jokingbear/research-idea',
    license='MIT',
    author='jokingbear',
)
