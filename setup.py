from setuptools import setup

setup(
    name='plasma',
    version='1.8.8',
    packages=['plasma', 'plasma.meta', 'plasma.logging', 'plasma.modules', 'plasma.training', 'plasma.training.data',
              'plasma.functional', 'plasma.huggingface', 'plasma.search_engines', 'plasma.parallel_processing'],
    url='https://github.com/jokingbear/research-idea',
    license='MIT',
    author='jokingbear',
)
