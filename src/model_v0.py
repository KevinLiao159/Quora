"""
model v0: baseline linear model
NBSVM (Naive Bayes - Support Vector Machine)
Youtube link: https://www.youtube.com/watch?v=37sFIak42Sc&feature=youtu.be&t=3745   # noqa

features: basic naive bayes features from count-based or tfidf
model: SVM, or sklearn logistic regression (faster)
"""

import nlp
import operator
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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

    def train(self, X_train, y_train, X_val, y_val, Cs=None):
        """
        trainer to score auc over a grid of Cs

        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets

        Cs: list of floats | int

        Return
        ------
        self
        """
        # init grid
        origin_C = self.C
        if Cs is None:
            Cs = [0.01, 0.1, 0.5, 1, 2, 10]
        # score
        scores = {}
        for C in Cs:
            # fit
            self.C = C
            model = self.fit(X_train, y_train)
            # predict
            y_proba = model.predict_proba(X_val)
            scores[C] = metrics.roc_auc_score(y_val, y_proba)
            print("Val AUC Score: {:.4f} with C = {}".format(scores[C], C)) # noqa
        # get max
        self._best_C, self._best_score = max(scores.items(), key=operator.itemgetter(1))  # noqa
        # reset
        self.C = origin_C
        return self

    @property
    def best_param(self):
        check_is_fitted(self, ['_clf'])
        return self._best_C

    @property
    def best_score(self):
        check_is_fitted(self, ['_clf'])
        return self._best_score


def get_model():
    return NbSvmClassifier()


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
        return nlp.word_tokenize(text, remove_punct=False, remove_num=True)

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
        return nlp.char_tokenize(text, remove_punct=False, remove_num=True)

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
    return sparse.hstack([word_transformer(df_text), char_transformer(df_text)])    # noqa
