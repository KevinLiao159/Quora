import os
import time
import re
import string
import unicodedata
import pandas as pd

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union


"""
text cleaning
"""


def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤$&#‘’])')
    return re_tok.sub(r' \1 ', text)


def remove_punctuation(text):
    """
    remove punctuation from text
    """
    re_tok = re.compile(f'([{string.punctuation}])')
    return re_tok.sub(' ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]{1,})')
    return re_tok.sub(r' \1 ', text)


def decontracted(text):
    """
    de-contract the contraction
    """
    # specific
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def clean_numbers(text):
    """
    replace number with hash
    """
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def preprocess(text, remove_punct=False):
    """
    preprocess text into clean text for tokenization
    """
    # 1. normalize
    text = normalize_unicode(text)
    # 2. to lower
    text = text.lower()
    # 3. space
    text = spacing_punctuation(text)
    text = spacing_number(text)
    # (optional)
    if remove_punct:
        text = remove_punctuation(text)
    # 4. de-contract
    text = decontracted(text)
    # 5. clean number
    text = clean_numbers(text)
    return text


def tokenize(text, remove_punct=False):
    """
    tokenize text into list of tokens
    """
    # 1. preprocess
    text = preprocess(text, remove_punct)
    # 2. tokenize
    tokens = text.split()
    return tokens


def preprocessor(text):
    return preprocess(text, remove_punct=False)


def tokenizer(text):
    return tokenize(text, remove_punct=False)


"""
model
"""


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes - Support Vector Machine
    """
    def __init__(self, C=0.8, dual=True, n_jobs=-1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(X.multiply(self._r))

    def predict_proba(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(X.multiply(self._r))[:, 1]

    def fit(self, X, y):
        # Check that X and y have correct shape
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=True)

        def pr(X, y_i, y):
            p = X[y == y_i].sum(0)
            return (p+1) / ((y == y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(X, 1, y) / pr(X, 0, y)))
        X_nb = X.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C,
            dual=self.dual,
            n_jobs=self.n_jobs
        ).fit(X_nb, y)
        return self


def get_model():
    return NbSvmClassifier()


def transform(df_text, stop_words=None, add_char=True):
    """
    Tf-idf transform and extract features from raw text dataframe

    Parameters
    ----------
    df_text: dataframe, single column with text

    stop_words: string {‘english’}, list, or None (default)

    add_char: bool, add n-grams char features

    Return
    ------
    df_features
    """
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        ngram_range=(1, 3),
        tokenizer=tokenizer, analyzer='word',
        min_df=3, max_df=0.9, max_features=None,
        use_idf=True, smooth_idf=True, sublinear_tf=True,
        stop_words=stop_words)
    if add_char:
        char_vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            ngram_range=(1, 3),
            preprocessor=preprocessor, analyzer='char',
            min_df=3, max_df=0.9, max_features=None,
            use_idf=True, smooth_idf=True, sublinear_tf=True,
            stop_words=stop_words)
        vectorizer = make_union(vectorizer, char_vectorizer)
    return vectorizer.fit_transform(df_text)


def load_and_preprocess(datapath):
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
    t0 = time.time()
    print("Loading data")
    df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
    df_test = pd.read_csv(os.path.join(datapath, "test.csv"))
    train_test_cut = df_train.shape[0]
    print("Train data with shape : ", df_train.shape)
    print("Test data with shape : ", df_test.shape)
    # concat text data into single dataframe
    df_all = pd.concat(
        [df_train[['question_text']], df_test[['question_text']]],
        axis=0).reset_index(drop=True)
    # transform
    X_features = transform(df_all['question_text'])
    X_train = X_features[:train_test_cut]
    X_test = X_features[train_test_cut:]
    print('Load and preprocessing took {:.2f}s'.format(time.time() - t0))
    return df_train, df_test, X_train, X_test


def create_submission(X_train, y_train, X_test, df_test, thres,
                      filepath='submission.csv'):
    """
    train model with entire training data, predict test data,
    and create submission file

    Parameters
    ----------
    X_train, y_train, X_test: features and targets

    df_test: dataframe, test data

    thres: float, a decision threshold for classification

    filepath: tmp path to store score csv
    """
    # get model
    model = get_model()
    # train model
    t0 = time.time()
    print('Start to train model')
    model = model.fit(X_train, y_train)
    print('Training took {:.2f}'.format(time.time() - t0))
    # predict
    print('Start to predict')
    y_pred = (model.predict_proba(X_test) > thres).astype('int')
    # create submission file
    print('Save submission file to {}'.format(filepath))
    pd.DataFrame(
        {
            'qid': df_test.qid,
            'prediction': y_pred
        }
    ).to_csv(filepath, index=False)


if __name__ == '__main__':
    # config
    # SHUFFLE = True
    DATA_PATH = '../input/'
    THRES = 0.24

    t0 = time.time()
    # 1. load and preprocess data
    df_train, df_test, X_train, X_test = load_and_preprocess(DATA_PATH)
    # 2. create submission file
    create_submission(X_train, df_train.target, X_test, df_test, THRES)
    # record time spent
    print('All done and it took {:.2f}s'.format(time.time() - t0))
