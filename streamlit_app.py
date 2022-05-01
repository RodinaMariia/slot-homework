import requests
import pandas as pd
import streamlit as st
import time
import random
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from numpy import number
from tools import load_data


def get_stats() -> dict:
    """Load statistics from API"""
    return requests.get('https://slot-ml.com/api/v1/users/4e5bd33034c6cf34acd47a679ed113f367c21d1c/stats').json()


def get_data() -> dict:
    """Push clusters information to API """
    return requests.get(
        'https://slot-ml.com/api/v1/users/4e5bd33034c6cf34acd47a679ed113f367c21d1c/vectors/?random').json()


def create_clusters(data: pd.DataFrame) -> pd.DataFrame:
    """Split data to several clusters"""
    cls = SpectralClustering(35, n_init=5,
                             assign_labels='discretize',
                             affinity='nearest_neighbors',
                             random_state=0)
    return pd.DataFrame(cls.fit_predict(data), columns=['pred'], index=data.index)


def create_classifier(data: pd.DataFrame) -> RandomForestClassifier:
    train_data = data.set_index('id').select_dtypes(include=[number])
    labels = create_clusters(train_data)
    clf = RandomForestClassifier(max_depth=3, n_estimators=10, random_state=0)
    clf.fit(train_data, labels['pred'])
    return clf


def get_prediction(data: pd.DataFrame, clf: RandomForestClassifier):
    """Count probabilities of every class and return the max of ot."""
    preds = clf.predict_proba(data.set_index('id').select_dtypes(include=[number]))
    preds = pd.DataFrame(preds, index=data.id)
    return preds.idxmax(axis=1)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_pretrained():
    """Loads prepared data and preprocessor"""
    return load_data()


def run_test_app():
    """
    User interface working with test data. Take some information from the existing raw data,
    count clusters and show statistical information from the main server.

    ..note:: Do not obtain any new data from the main server.
    """
    UPDATING_PERIOD = 120
    USE_CLUSTERS = False

    with st.spinner("Loading pretrained models"):
        #
        data, preprocessor = load_pretrained()
        raw_data = pd.read_csv('data/raw_data.csv', dtype=object)
        if not USE_CLUSTERS:
            clf = create_classifier(data)

    st.header('New data stored in cluster:')
    st.caption('New information loads every {} sec and includes in the cluster.'.format(UPDATING_PERIOD))
    stats = get_stats()
    main_table = st.table()
    accuracy_title = st.subheader('During accuracy is {}'.format(stats.get('stats')[0].get('avg_accuracy')))
    stats_json = st.json(stats)
    while True:
        with st.spinner("Preprocess data..."):
            data = data.fillna(0)
            data_upd_raw = raw_data.iloc[[random.randrange(raw_data.shape[0])], :]
            data_upd = preprocessor.transform(data_upd_raw)
            data = pd.concat([data, data_upd])

        if USE_CLUSTERS:
            preds = create_clusters(data.set_index('id').select_dtypes(include=[number]))
        else:
            preds = get_prediction(data_upd, clf)

        main_table.write(preds.loc[preds.index.isin(data_upd_raw.id)].drop_duplicates())
        stats = get_stats()
        accuracy_title.write('New accuracy is {}'.format(stats.get('stats')[0].get('avg_accuracy')))
        stats_json.write(stats)

        time.sleep(UPDATING_PERIOD)


if __name__ == '__main__':
    run_test_app()
