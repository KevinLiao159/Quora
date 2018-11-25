"""
baseline model:
NBSVM (Naive Bayes - Support Vector Machine)
Youtube link: https://www.youtube.com/watch?v=37sFIak42Sc&feature=youtu.be&t=3745   # noqa

features: basic naive bayes features from count-based or tfidf
model: SVM, or sklearn logistic regression (faster)
"""

import nlp
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes - Support Vector Machine
    """
    def __init__(self, C=4.0, dual=True, n_jobs=-1):
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
        return self._clf.predict_proba(X.multiply(self._r))

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


def tokenizer(text):
    return nlp.tokenize(text, remove_punct=True)


def transform(df_text, tfidf=True):
    """
    transform and extract features from raw text dataframe

    Parameters
    ----------
    df_text: dataframe, single column with text

    Return
    ------
    df_features
    """
    if tfidf:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            tokenizer=tokenizer,
            min_df=3, max_df=0.9,
            strip_accents='unicode',
            use_idf=1, smooth_idf=1,
            sublinear_tf=1)
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            tokenizer=tokenizer,
            min_df=3, max_df=0.9,
            strip_accents='unicode',
            binary=True)
    return vectorizer.fit_transform(df_text)
