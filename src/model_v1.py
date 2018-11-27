"""
model v1: baseline tree model
Light GBM
Doc: https://lightgbm.readthedocs.io/en/latest/Python-API.html

features: count-based (categorical) or tfidf (weights)
model: Light GBM - DART and GBDT with different seeds
"""

import gc
import model_v0
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import lightgbm


class LightgbmClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM classifier for my own interface - sklearn like
    """
    def __init__(self, params,
                 feature_name='auto',
                 categorical_feature='auto'):
        """
        Parameter
        ---------
        params: dict, parameters for training

        feature_name: list of strings or 'auto', optional (default="auto")

        categorical_feature: list of strings or int, or 'auto',
            optional (default="auto")
        """
        self.params = self.params
        self.feature_name = feature_name
        self.categorical_feature = self.categorical_feature

    def predict(self, X, num_iteration=None):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        return (self._clf.predict(X, num_iteration=None) > 0.5).astype(int)

    def predict_proba(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(X, num_iteration=None)

    def get_dataset(self, X, y, free_raw_data=True):
        """
        convert data into lightgbm consumable format

        Parameters
        ----------
        X: string, numpy array, pandas DataFrame, scipy.sparse or
            list of numpy arrays

        y: list, numpy 1-D array, pandas Series / one-column DataFrame \
            or None, optional (default=None)

        free_raw_data: bool, optional (default=True)

        Return
        ------
        lightgbm dataset
        """
        return lightgbm.Dataset(
            data=X, label=y,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            free_raw_data=free_raw_data)

    def train(self, X_train, y_train, X_val, y_val,
              num_boost_round=100,
              early_stopping_rounds=20,
              verbose_eval=True):
        """
        train lightgbm and monitor the best iteration with validation

        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets

        num_boost_round: int, optional (default=100),
            number of boosting iterations

        early_stopping_rounds: int or None, optional (default=None)),
            activates early stopping. The model will train until the \
            validation score stops improving

        verbose_eval: bool or int, optional (default=True)
        """
        # Check that X and y have correct shape
        y_train, y_val = y_train.values, y_val.values
        X_train, y_train = check_X_y(X_train, y_train, accept_sparse=True)
        X_val, y_val = check_X_y(X_val, y_val, accept_sparse=True)
        # prep datasets
        train_set = self.get_dataset(
            X_train, y_train,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            free_raw_data=True)
        valid_set = self.get_dataset(
            X_val, y_val,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            free_raw_data=True)
        del X_train, y_train, X_val, y_val
        gc.collect()
        # train
        self._clf = lightgbm.train(
            params=self.params,
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            verbose_eval=verbose_eval)

    def fit(self, X, y,
            best_iteration=100):
        """
        fit lightgbm with best iteration, which is the best model

        Parameters
        ----------
        X, y: features and targets

        best_iteration: int, optional (default=100),
            number of boosting iterations
        """
        # Check that X and y have correct shape
        y = y.values
        X, y = check_X_y(X, y, accept_sparse=True)
        # prep datasets
        train_set = self.get_dataset(
            X, y,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            free_raw_data=True)
        del X, y
        gc.collect()
        # train
        self._clf = lightgbm.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=best_iteration,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature)


def get_model():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.3,
        'num_leaves': 12,
        'max_depth': 2,
        'min_split_gain': 0,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'min_child_samples': 1000,
        'min_child_weight': 0,
        'max_bin': 100,
        'subsample_for_bin': 200000,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'scale_pos_weight': 5,
        'metric': 'auc',
        'nthread': 2,
        'verbose': 0
    }
    return LightgbmClassifier(params)


def transform(df_text, tfidf=True, stop_words=None, add_char=True):
    return model_v0.transform(df_text, tfidf, stop_words, add_char)
