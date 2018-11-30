"""
Weighted averaging model predictions
"""

# import gc
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils.validation import check_X_y, check_is_fitted


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    """
    weighted average prediction from list of modules
    """
    def __init__(self, modules, weights):
        self.modules = modules
        self.weights = weights

    def prediction(self, X):
        pass


def get_model():
    pass


def transform(df_text):
    pass
