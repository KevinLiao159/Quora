import os
import time
import pandas as pd
from contextlib import contextmanager


def load_and_preprocess(datapath, module):
    """
    load and preprocess
    Parameters
    ----------
    datapath: str, data directory that contains train.csv and test.csv
    module: a python module
    Returns
    -------
    df_train, df_test: dataframe with raw text
    X_train, X_test: matrix with proper features
    """
    print("loading data ......")
    df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
    df_test = pd.read_csv(os.path.join(datapath, "test.csv"))
    train_test_cut = df_train.shape[0]
    print("train data with shape : ", df_train.shape)
    print("test data with shape : ", df_test.shape)
    # concat text data into single dataframe
    df_all = pd.concat(
        [df_train[['question_text']], df_test[['question_text']]],
        axis=0).reset_index(drop=True)
    # transform
    X_features = module.transform(df_all['question_text'])
    # handle multiple inputs
    if isinstance(X_features, list):
        X_train = [X[:train_test_cut] for X in X_features]
        X_test = [X[train_test_cut:] for X in X_features]
    # single input
    else:
        X_train = X_features[:train_test_cut]
        X_test = X_features[train_test_cut:]
    return df_train, df_test, X_train, X_test


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_trained_model(model, weights_path):
    model.load_weights(weights_path)
    return model
