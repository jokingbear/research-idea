import pandas as pd


class ObjectInquirer:

    def __init__(self, obj):
        self.original_object = obj

        self._registered_obj_getter = {
            dict: _identity,
            list: _identity,
            pd.DataFrame: _dataframe,
            pd.Series: _series
        }
    
    def register_getter(self, obj_type:type, func):
        self._registered_obj_getter[obj_type] = func

    def __getitem__(self, path):
        obj_type = type(self.original_object)
        getter = self._registered_obj_getter.get(obj_type, None)
        if getter is None:
            raise NotImplementedError(f'there is no key getter for object of type {obj_type}, please call register function')

        data = getter(self.original_object)

        if isinstance(path, list):
            obj = data[path[0]]

            if len(path) == 1:
                return obj
            else:
                return type(self)(obj)[path[1:]]
        else:
            return data[path]

    def __repr__(self) -> str:
        return repr(self.original_object)



def _identity(obj):
    return obj


def _dataframe(df:pd.DataFrame):
    return df.iloc


def _series(s:pd.Series):
    return s.loc
