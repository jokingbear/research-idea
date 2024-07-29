import pandas as pd


class PathInquirer:

    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, path):
        data = self._cast_data(self.data)

        if isinstance(path, list):
            obj = data[path[0]]

            if len(path) == 1:
                return obj
            else:
                return type(self)(obj)[path[1:]]
        else:
            return data[path]

    def _cast_data(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.iloc
        elif isinstance(data, pd.Series):
            data = data.loc
        
        return data

    def __repr__(self) -> str:
        return repr(self.data)
