import pandas as pd


class ObjectInquirer:

    def __init__(self, obj):
        self.original_object = obj

        self._registered_obj_getter = {
            dict: _item_getter,
            list: _item_getter,
            pd.DataFrame: _dataframe_iloc,
            pd.Series: _series_loc
        }
    
    def register_getter(self, obj_type:type, func):
        self._registered_obj_getter[obj_type] = func

    def __getitem__(self, path):
        if isinstance(path, list):
            current_key = path[0]
            next_path = path[1:]
        else:
            current_key = path
            next_path = []
        
        obj_type = type(self.original_object)
        getter = self._registered_obj_getter.get(obj_type, getattr)
      
        value = getter(self.original_object, current_key)

        if len(next_path) > 0:
            new_inquirer = ObjectInquirer(value)
            new_inquirer._registered_obj_getter = self._registered_obj_getter
            return new_inquirer[next_path]

        return value

    def __repr__(self) -> str:
        return repr(self.original_object)


def _item_getter(obj, key):
    return obj[key]


def _dataframe_iloc(df:pd.DataFrame, idx):
    return df.iloc[idx]


def _series_loc(s:pd.Series, key):
    return s.loc[key]
