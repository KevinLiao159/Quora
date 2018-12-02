"""
Neural Network Trainer
"""
import operator
import numpy as np
from sklearn import utils
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


class NeuralNetworkClassifier:
    """
    Neural Network classifier for my own interface - sklearn like
    """
    def __init__(self, model, batch_size=512, epochs=10,
                 balancing_class_weight=False, filepath=None):
        """
        Parameter
        ---------
        model: Keras model

        batch_size: int or None, number of samples per gradient update

        epochs: int, number of epochs to train the model

        balancing_class_weight: bool, if True, uses the values of y to
            automatically adjust weights inversely proportional to
            class frequencies in the input data as
            n_samples / (n_classes * np.bincount(y))

        filepath: str, data directory that stores model pickle
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.balancing_class_weight = balancing_class_weight
        self.filepath = filepath
        # compile model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=[precision_score, recall_score, f1_score])

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X):
        return self.model.predict(X)

    def train(self, X_train, y_train, X_val, y_val, verbose=1):
        """
        train neural network and monitor the best iteration with validation

        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets

        verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch

        Return
        ------
        self
        """
        callbacks = []
        # get callbacks
        callbacks.append(
            EarlyStopping(
                monitor='val_f1_score',
                patience=3,
                verbose=verbose
            )
        )
        if self.filepath:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.filepath,
                    monitor='val_f1_score',
                    save_best_only=True,
                    save_weights_only=True
                )
            )
        # get class_weight
        if self.balancing_class_weight:
            class_weight = utils.class_weight.compute_class_weight(
                'balanced', np.unique(y_train), y_train)
        else:
            class_weight = None
        # train model
        self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(X_val, y_val),
            shuffle=True,
            class_weight=class_weight)
        return self

    def fit(self, X, y, best_iteration=2, verbose=1):
        """
        fit lightgbm with best iteration, which is the best model

        Parameters
        ----------
        X, y: features and targets

        best_iteration: int, optional (default=100),
            number of boosting iterations

        verbose: int, 0 = silent, 1 = progress bar, 2 = one line per epoch

        Return
        ------
        self
        """
        # get class_weight
        if self.balancing_class_weight:
            class_weight = utils.class_weight.compute_class_weight(
                'balanced', np.unique(y), y)
        else:
            class_weight = None
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=best_iteration,
            verbose=verbose,
            shuffle=True,
            class_weight=class_weight)
        return self

    @property
    def best_param(self):
        scores = self.model.history.history['val_f1_score']
        best_iteration, _ = max(enumerate(scores), key=operator.itemgetter(1))
        return best_iteration + 1

    @property
    def best_score(self):
        scores = self.model.history.history['val_f1_score']
        _, best_val_f1 = max(enumerate(scores), key=operator.itemgetter(1))
        return best_val_f1


def recall_score(y_true, y_proba, thres=0.5):
    """
    Recall metric

    Only computes a batch-wise average of recall

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected
    """
    # get prediction
    y_pred = K.cast(K.greater(y_proba, thres), dtype='float32')
    # calc
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_proba, thres=0.5):
    """
    Precision metric

    Only computes a batch-wise average of precision

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant
    """
    # get prediction
    y_pred = K.cast(K.greater(y_proba, thres), dtype='float32')
    # calc
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_proba, thres=0.5):
    """
    F1 metric: geometric mean of precision and recall
    """
    precision = precision_score(y_true, y_proba, thres)
    recall = recall_score(y_true, y_proba, thres)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
