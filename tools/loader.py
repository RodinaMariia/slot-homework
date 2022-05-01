"""
Tools for working with saved data.
"""
import os
import pandas as pd
import pickle
from typing import Tuple
import tools.preprocessing as preprocessing


def create_empty_data() -> pd.DataFrame:
    """Create empty dataframe for keeping incoming information."""
    return pd.DataFrame({'meta1': pd.Series(dtype='str'),
                         'id': pd.Series(dtype='str'),
                         'meta2': pd.Series(dtype='str'),
                         'vector': pd.Series(dtype='str'),
                         'meta3': pd.Series(dtype='int'),
                         'meta6': pd.Series(dtype='str'),
                         'meta4': pd.Series(dtype='int'),
                         'meta5': pd.Series(dtype='str'),
                         }
                        )


def load_data() -> Tuple[pd.DataFrame, preprocessing.Preprocessor]:
    """Return previously saved data and pretrained Preprocessor transformer."""
    try:
        data = pd.read_csv('data/data.csv')
    except Exception as e:
        print(e)
        data = create_empty_data()
        data.to_csv('data/data.csv', index=False)
    try:
        preprocessor = load_pickle('preprocessor')
    except Exception as e:
        print(e)
        preprocessor = preprocessing.Preprocessor(preprocessing.create_handler())
        preprocessor.fit_transform(pd.read_csv('data/raw_data.csv', dtype=object))
        save_pickle(preprocessor, 'preprocessor')
    return data, preprocessor


def save_pickle(file, name):
    """Util for saving pickle-files"""
    with open(os.path.join('data/', name + '.pickle'), 'wb') as f:
        pickle.dump(file, f)


def load_pickle(name: str):
    """Util for loading pickle-files"""
    with open(os.path.join('data/', name + '.pickle'), 'rb') as f:
        return pickle.load(f)
