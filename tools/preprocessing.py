"""
Contains:
    Class converts raw data to a numerical representation.
    Set of rules that helps to find dangerous and safe rows in the raw data.
"""
import collections
import itertools
import ipinfo
import math
import category_encoders as ce
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.models.fasttext import FastText
from ua_parser import user_agent_parser


class Watchman:
    """
    Simple classifier to find dangerous and save data records by several analytical rules.

    Attributes:
        df: pd.DataFrame,
            Analysing dataset.

        white_list: list[str],
            Allowed web server scanners.

    """

    def __init__(self, data):
        self.df = data.copy()
        self.white_list = ['arachni', 'nikto', 'openvas', 'acunetix']
        self._preprocess()

    def _preprocess(self):
        self.df['meta4'] = self.df['meta4'].astype(float)
        self.df['meta3'] = self.df['meta3'].astype(float)
        self.df['meta5'] = self.df['meta5'].fillna('')
        self.df['meta2'] = self.df['meta2'].fillna('')
        self.df['vector'] = self.df.vector.str.lower()
        self.df['meta5'] = self.df.meta5.str.lower()
        for idx, row in self.df.iterrows():
            self.df.loc[idx, 'meta2'] = self.df.loc[idx, 'meta2'].replace(self.df.loc[idx, 'meta1'] + '.', '')
        self.df['len_meta2'] = self.df['meta2'].apply(len)

    def white_list(self, func):
        """Filter alloweded bots actions. """

        def wrapper(*args, **kwargs):
            black_list_edt = []
            for one_id in func(self, *args, **kwargs):
                res = [1 for one_word in self.white_list if
                       self.df.loc[self.df.id == one_id, 'meta5'].str.contains(one_word).any() |
                       self.df.loc[self.df.id == one_id, 'vector'].str.contains(one_word).any()]

                if sum(res) == 0:
                    black_list_edt.append(one_id)
            return black_list_edt

        return wrapper

    @white_list
    def get_bl_by_request_stops(self) -> list:
        black_list = []
        black_words = ['windows', 'ini', 'net', 'select', 'delete',
                       'update', 'password', ' or ']
        for one_word in black_words:
            black_list.extend(self.df.loc[self.df.vector.str.contains(one_word), 'id'].to_list())
        return black_list

    @white_list
    def get_bl_by_data_volume(self, q=0.95) -> list:
        return self.df.loc[self.df.meta4 >= self.df.meta4.quantile(q=q), 'id'].to_list()

    @white_list
    def get_bl_by_request_frequency(self, threshold=3) -> list:
        black_ip = self.df.meta6.value_counts()[self.df.meta6.value_counts() > threshold].index.to_list()
        return self.df.loc[self.df.meta6.isin(black_ip), 'id'].to_list()

    @white_list
    def get_bl_by_user_agent_stops(self) -> list:
        black_list = []
        black_words = ['apache-httpclient', 'python', 'winhttprequest']
        for one_word in black_words:
            black_list.extend(self.df.loc[self.df.meta5.str.contains(one_word), 'id'].to_list())
        return black_list

    @white_list
    def get_bl_by_user_agent_frequency(self, threshold=0.05) -> list:
        white_words = ['arachni', 'openvas', 'mozilla', 'ucweb', 'ucbrowser',
                       'firefox', 'chrome', 'safari']
        vecs = self.df['meta5'].apply(simple_preprocess)
        dictionary = corpora.Dictionary(vecs)
        rare_ids = [key for key, value in dictionary.dfs.items() if value < round(self.df.shape[0] * threshold)]
        dictionary.filter_tokens(rare_ids)
        dictionary.add_documents([white_words])
        self.df['meta5_cls'] = [dictionary.doc2bow(one_vec) for one_vec in vecs]
        return self.df.loc[self.df['meta5_cls'].apply(lambda x: len(x) == 0), 'id'].to_list()

    @white_list
    def get_bl_by_request_meta_regex(self) -> list:
        black_list = []
        sliced_df = self.df.loc[(self.df.meta1 == 'REQUEST_HEADERS') &
                                (self.df.meta2 == 'Referer')]
        black_list.extend(sliced_df.loc[sliced_df.vector.str.contains('file'), 'id'].to_list())

        sliced_df = self.df.loc[(self.df.meta1 == 'REQUEST_HEADERS') &
                                (self.df.meta2 == 'Connection')]
        black_list.extend(sliced_df.loc[~sliced_df.vector.str.contains('close|alive', regex=True), 'id'].to_list())

        sliced_df = self.df.loc[(self.df.meta1 == 'REQUEST_HEADERS') &
                                (self.df.meta2 == 'Connection')]
        black_list.extend(sliced_df.loc[~sliced_df.vector.str.contains('close|alive', regex=True), 'id'].to_list())

        sliced_df = self.df.loc[(self.df.meta1 == 'REQUEST_HEADERS') &
                                (self.df.meta2 == 'X-Forwarded-For')]
        black_list.extend(sliced_df.loc[sliced_df.vector.str.contains('[^a-z0-9.:]', regex=True), 'id'].to_list())
        return black_list

    @white_list
    def get_bl_by_request_meta_frequency(self, q=0.95) -> list:
        sliced_df = self.df.loc[(self.df.meta1.isin(['REQUEST_POST_ARGS',
                                                     'REQUEST_GET_ARGS', 'REQUEST_COOKIES',
                                                     'RESPONSE_HEADERS', 'REQUEST_HEADERS']))]
        return sliced_df.loc[sliced_df.len_meta2 >= self.df.len_meta2.quantile(q=q), 'id'].to_list()

    @white_list
    def get_bl_full_rules(self, q=0.95, threshold_request=3) -> list:
        # Фильтрация построена неэффективно, нужно обдумать, как разграничить использование белого списка.
        black_list = []
        black_list.extend(self.get_bl_by_request_stops())
        black_list.extend(self.get_bl_by_data_volume(q))
        black_list.extend(self.get_bl_by_request_frequency(threshold_request))
        black_list.extend(self.get_bl_by_user_agent_stops())
        black_list.extend(self.get_bl_by_request_meta_regex())
        black_list.extend(self.get_bl_by_request_meta_frequency(q))

        black_ip = self.df.loc[self.df.id.isin(black_list), 'meta6']
        black_list.extend(self.df.loc[self.df.meta6.isin(black_ip), 'id'].to_list())
        return list(set(black_list))

    def get_wt_list(self) -> list:
        """Return list of permitted actions"""
        white_list_edt = []
        for one_id in self.df.id:
            res = [1 for one_word in self.white_list if
                   self.df.loc[self.df.id == one_id, 'meta5'].str.contains(one_word).any() |
                   self.df.loc[self.df.id == one_id, 'vector'].str.contains(one_word).any()]

            if sum(res) > 0:
                white_list_edt.append(one_id)
        return white_list_edt


class Preprocessor:
    """
    Transformer that converts raw data to a numerical representation and a little bit more.

    Attributes:
        ip_cols: pd.list,
            Columns from the ip parser.

        stat_cols: pd.list,
            Columns with the raw texts.

        cat_cols: pd.list,
            Columns with the categorical data.

        stat_funcs: pd.list,
            List of functions applied to texts.

        encoders: TransformerMixin,
            Categorical encoder, the one.

        vectorizers: dict,
            Column names with its FastText encoders.

        handler:
            IP parser handler.

    """

    def __init__(self, handler):
        self.ip_cols = ['hostname', 'country', 'latitude', 'longitude', 'org']
        self.stat_cols = ['vector', 'hostname', 'meta2']
        self.cat_cols = ['meta1', 'device', 'os', 'user_agent'] + self.ip_cols
        self.cat_cols.remove('hostname')
        self.stat_funcs = {'letters': calculate_letters, 'digits': calculate_digits,
                           'non_letters': calculate_non_letters, 'entropy': calculate_entropy,
                           }
        self.encoders = None
        self.vectorizers = None
        self.handler = handler

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._basic_transform(data)
        self._fit_all_transformers(df)
        return self._transformers_transform(df).fillna(0)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._basic_transform(data)
        return self._transformers_transform(df).fillna(0)

    def _basic_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df.fillna('')
        df['vector'] = df.vector.str.lower()
        df['meta5'] = df.meta5.str.lower()
        df['meta4'] = df['meta4'].astype(int)
        df['meta3'] = df['meta3'].astype(int)
        df['len_vec'] = df['vector'].apply(len)

        for one_col in ['device', 'os', 'user_agent', 'hostname',
                        'country', 'latitude', 'longitude', 'org']:
            df[one_col] = ''
        for one_pair in itertools.product(self.stat_cols, list(self.stat_funcs.keys())):
            df['{}_{}'.format(*one_pair)] = 0

        parsed_meta5_vec = []
        for idx, row in df.iterrows():
            #User-agent parser
            parsed_meta5 = user_agent_parser.Parse(df.loc[idx, 'meta5'])
            df.loc[idx, 'device'] = parsed_meta5.get('device').get('family')
            df.loc[idx, 'os'] = parsed_meta5.get('os').get('family')
            df.loc[idx, 'user_agent'] = parsed_meta5.get('user_agent').get('family')
            vec = list(itertools.chain(list(parsed_meta5.get('device').values()),
                                       list(parsed_meta5.get('os').values()),
                                       list(parsed_meta5.get('user_agent').values()),
                                       ))
            parsed_meta5_vec.append(vec)

            #IP parser
            try:
                details = self.handler.getDetails(df.loc[idx, 'meta6']).all
                for col in self.ip_cols:
                    df.loc[idx, col] = details.get(col) if details.get(col) else 'Empty'
            except Exception as e:
                for col in self.ip_cols:
                    df.loc[idx, col] = 'Error'

            # Statistical data
            for one_pair in itertools.product(self.stat_cols, self.stat_funcs.items()):
                df.loc[idx, '{}_{}'.format(one_pair[0], one_pair[1][0])] = one_pair[1][1](df.loc[idx, one_pair[0]])

        df['len_meta2'] = df['meta2'].apply(len)
        return df

    def _fit_all_transformers(self, data: pd.DataFrame):
        """Create and fit inner data transformers"""
        self.vectorizers = {one_col: self._create_tf_model(data, one_col) for one_col in self.stat_cols}
        self.encoders = ce.HashingEncoder(cols=self.cat_cols, n_components=8)
        self.encoders.fit(data)

    def _transformers_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies pretrained transformers"""
        data = self.encoders.transform(data)
        for key, value in self.vectorizers.items():
            vec = data.loc[:, key].apply(lambda x: self._get_sequence_vec(value, x))
            col_names = ['{}_{}'.format(key, i) for i in range(len(vec.iloc[0]))]
            data.loc[:, col_names] = vec.to_list()
        return data

    def _create_tf_model(self, data: pd.DataFrame, target_column: str, vector_size: int = 3) -> FastText:
        """Create FastText model to the one column"""
        target_data = list(map(simple_preprocess, data[target_column].apply(lambda x: str(x).replace('_', ' '))))
        model = FastText(vector_size=vector_size)
        model.build_vocab(corpus_iterable=target_data, min_count=1)
        model.train(
            corpus_iterable=target_data,
            epochs=model.epochs,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words
        )
        return model

    def _get_sequence_vec(self, model: FastText, text: str) -> list:
        """Create embeddings"""
        target_data = simple_preprocess(str(text).replace('_', ' '))
        vec = []
        for one_word in target_data:
            one_word = 'None' if one_word is None else one_word
            vec.append(model.wv[one_word])
        return np.mean(np.array(vec), axis=0) if vec else [0, 0, 0]


def calculate_letters(input_str: str) -> float:
    return sum(c.isalpha() for c in input_str)


def calculate_digits(input_str: str) -> float:
    return sum(c.isdigit() for c in input_str)


def calculate_non_letters(input_str: str) -> float:
    return sum((not c.isalpha()) for c in input_str)


def calculate_entropy(input_str: str) -> float:
    return (-1) * sum(
        i / len(input_str) * math.log2(i / len(input_str))
        for i in collections.Counter(input_str).values())


def create_handler():
    ACCESS_TOCKEN = '0000000000' #Put here correct tocken number.
    return ipinfo.getHandler(ACCESS_TOCKEN)
