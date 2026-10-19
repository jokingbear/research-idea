from collections.abc import Sequence
from sqlite3 import Connection

from ...functional import ReadableClass


class Statement(ReadableClass):
    
    def __init__(self, 
            text:str, params:Sequence,
            keyword_params:dict, 
            execute_many:bool
        ):
        super().__init__()        
        assert len(params) == 0 or len(keyword_params) == 0, 'params and keyword params are mutually exclusive'
        
        self.text = text
        self.params = params
        self.keyword_params = keyword_params
        self.many_strategy = execute_many
    
    def parameterize(self, *params, execute_many:bool|None=False, **keyword_params):
        execute_many = self.many_strategy if execute_many is None else execute_many
        return Statement(self.text, params, keyword_params, execute_many)

    def execute(self, connection:Connection):
        if self.many_strategy:
            return connection.executemany(self.text, self.params)
        else:
            return connection.execute(self.text, self.params)

    def _tree(self, tree):
        tree.add(f'statement\n{self.text.strip()}')
        
        if len(self.params) > 0:
            params = tree.add('params')
            params.add(f'{self.params[:5]}')

        return tree
