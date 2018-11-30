"""
Weighted averaging model predictions
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    """
    weighted average prediction from list of modules
    """
    def __init__(self, modules, weights):
        self.modules = modules
        self.weights = weights

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_clfs'])
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_clfs'])
        y_proba = np.zeros(len(X))
        for i, clf in enumerate(self.self._clfs):
            y_proba += clf.predict_proba(X).reshape(-1) * self.weights[i]
        return y_proba

    def fit(self, X, y):
        # Check that X and y have correct shape
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=True)
        # fit models
        self._clfs = []
        for module in self.modules:
            model = module.get_model()
            self._clfs.append(model.fit(X, y))
        return self


def get_model():
    import model_v0
    import model_v1
    # init weights
    weights = [0.5, 0.5]
    return BlendingClassifier(modules=[model_v0, model_v1], weights=weights)


def transform(df_text):
    import model_v1
    return model_v1.transform(df_text)
