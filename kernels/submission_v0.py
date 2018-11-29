import os
import time
import re
import string
import unicodedata
import pandas as pd
import numpy as np
from scipy import sparse
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression


"""
utils
"""


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


"""
text cleaning
"""


def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


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
    text = re.sub(r"(W|w)on\'t", "will not", text)
    text = re.sub(r"(C|c)an\'t", "can not", text)

    # general
    text = re.sub(r"(I|i)\'m", "i am", text)
    text = re.sub(r"(A|a)in\'t", "is not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    return text


def clean_number(text):
    """
    replace number with hash
    """
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text


def preprocess(text, remove_punct=False, remove_num=True):
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
    # 5. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = clean_number(text)
    # 6. remove space
    text = remove_space(text)
    return text


def word_tokenize(text, remove_punct=False, remove_num=True):
    """
    tokenize text into list of word tokens
    """
    # 1. preprocess
    text = preprocess(text, remove_punct, remove_num)
    # 2. tokenize
    tokens = text.split()
    return tokens


def char_tokenize(text, remove_punct=False, remove_num=True):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = word_tokenize(text, remove_punct, remove_num)
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


"""
transformer
"""


def word_transformer(df_text, stop_words=None):
    """
    transform and extract word features from raw text dataframe

    Parameters
    ----------
    df_text: dataframe, single column with text

    stop_words: string {‘english’}, list, or None (default)

    Return
    ------
    df_features
    """
    def _tokenizer(text):
        return word_tokenize(text, remove_punct=False, remove_num=True)

    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        ngram_range=(1, 3),
        tokenizer=_tokenizer,
        analyzer='word',
        min_df=3, max_df=0.9, max_features=None,
        use_idf=True, smooth_idf=True, sublinear_tf=True,
        stop_words=stop_words)
    return vectorizer.fit_transform(df_text)


def char_transformer(df_text, stop_words=None):
    """
    transform and extract word features from raw text dataframe

    Parameters
    ----------
    df_text: dataframe, single column with text

    stop_words: string {‘english’}, list, or None (default)

    Return
    ------
    df_features
    """
    def _tokenizer(text):
        return char_tokenize(text, remove_punct=False, remove_num=True)

    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        ngram_range=(1, 1),
        tokenizer=_tokenizer,
        analyzer='word',
        min_df=3, max_df=0.9, max_features=None,
        use_idf=True, smooth_idf=True, sublinear_tf=True,
        stop_words=stop_words)
    return vectorizer.fit_transform(df_text)


def transform(df_text):
    """
    transform and extract features from raw text dataframe

    Parameters
    ----------
    df_text: dataframe, single column with text

    Return
    ------
    features: dataframe, or numpy, scipy
    """
    return sparse.hstack([word_transformer(df_text), char_transformer(df_text)]).tocsr()    # noqa


"""
model
"""


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes - Support Vector Machine
    """
    def __init__(self, C=1.0, dual=True, n_jobs=-1):
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


def load_and_preprocess(datapath):
    """
    load and preprocess
    Parameters
    ----------
    datapath: str, data directory that contains train.csv and test.csv

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
    X_features = transform(df_all['question_text'])
    X_train = X_features[:train_test_cut]
    X_test = X_features[train_test_cut:]
    return df_train, df_test, X_train, X_test


def create_submission(X_train, y_train, X_test, df_test, thres):
    """
    train model with entire training data, predict test data,
    and create submission file

    Parameters
    ----------
    X_train, y_train, X_test: features and targets

    df_test: dataframe, test data

    thres: float, a decision threshold for classification

    module: a python module

    Return
    ------
    df_summission
    """
    # get model
    model = get_model()
    # train model
    print('fitting model')
    model = model.fit(X_train, y_train)
    # predict
    print('predicting probas')
    y_pred = (model.predict_proba(X_test) > thres).astype('int')
    # create submission file
    return pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred})


if __name__ == '__main__':
    # config
    # SHUFFLE = True
    DATA_PATH = '../input/'
    FILE_PATH = 'submission.csv'
    THRES = 0.23

    t0 = time.time()
    # 1. load and preprocess data
    with timer("Load and Preprocess"):
        df_train, df_test, X_train, X_test = load_and_preprocess(DATA_PATH)
    # 2. create submission file
    with timer('Trainning and Creating Submission'):
        df_submission = create_submission(
            X_train, df_train.target,
            X_test, df_test,
            THRES)
        df_submission.to_csv(FILE_PATH, index=False)
        print('Save submission file to {}'.format(FILE_PATH))
    # record time spent
    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0)) # noqa
