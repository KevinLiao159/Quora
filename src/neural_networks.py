"""
Neural Network Trainer
"""
import operator
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Activation, Wrapper
from keras.engine.topology import Layer
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)


# decision threshold
THRES = 0.35


class NeuralNetworkClassifier:
    """
    Neural Network classifier for my own interface - sklearn like
    """
    def __init__(self, model, batch_size=512, epochs=10, val_score='val_loss',
                 reduce_lr=True, balancing_class_weight=False, filepath=None):
        """
        Parameter
        ---------
        model: Keras model

        batch_size: int or None, number of samples per gradient update

        epochs: int, number of epochs to train the model

        val_score: str, score to monitor. ['accuracy', 'precision_score',
            'recall_score', 'f1_score', 'roc_auc_score']

        reduce_lr: bool, if True, add a Keras callback function that
            reduce learning rate when a metric has stopped improving

        balancing_class_weight: bool, if True, uses the values of y to
            automatically adjust weights inversely proportional to
            class frequencies in the input data as
            n_samples / (n_classes * np.bincount(y))

        filepath: str, data directory that stores model pickle
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_score = val_score
        self.reduce_lr = reduce_lr
        self.balancing_class_weight = balancing_class_weight
        self.filepath = filepath
        # compile model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision_score, recall_score,
                     f1_score, roc_auc_score])

    def _get_class_weight(self, y):
        # get class_weight
        if self.balancing_class_weight:
            from sklearn import utils
            return utils.class_weight.compute_class_weight(
                'balanced', np.unique(y), y)
        else:
            return None

    def _get_callbacks(self):
        callbacks = []
        # get callbacks
        callbacks.append(
            EarlyStopping(
                monitor=self.val_score,
                patience=2,
                verbose=1
            )
        )
        if self.filepath:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.filepath,
                    monitor=self.val_score,
                    save_best_only=True,
                    save_weights_only=True
                )
            )
        if self.reduce_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.val_score,
                    factor=0.6,
                    patience=1,
                    min_lr=0.0001,
                    verbose=2
                )
            )
        return callbacks

    def predict(self, X):
        return (self.predict_proba(X) > THRES).astype(int)

    def predict_proba(self, X):
        return self.model.predict([X], batch_size=1024, verbose=1)

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
        # get callbacks
        callbacks = self._get_callbacks()
        # get class_weight
        class_weight = self._get_class_weight(y_train)
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

    def fit(self, X, y, best_iteration=6, verbose=1):
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
        class_weight = self._get_class_weight(y)
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=best_iteration,
            verbose=verbose,
            shuffle=True,
            class_weight=class_weight)
        # save model
        if self.filepath:
            self.model.save_weights(self.filepath)
            print('saved fitted model to {}'.format(self.filepath))
        return self

    @property
    def best_param(self):
        scores = self.model.history.history[self.val_score]
        if 'loss' in self.val_score:
            func = min
        else:
            func = max
        best_iteration, _ = func(enumerate(scores), key=operator.itemgetter(1))
        return best_iteration + 1

    @property
    def best_score(self):
        scores = self.model.history.history[self.val_score]
        if 'loss' in self.val_score:
            func = min
        else:
            func = max
        _, best_val_f1 = func(enumerate(scores), key=operator.itemgetter(1))
        return best_val_f1


"""
customized metrics during model training
"""


def recall_score(y_true, y_proba, thres=THRES):
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


def precision_score(y_true, y_proba, thres=THRES):
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


def f1_score(y_true, y_proba, thres=THRES):
    """
    F1 metric: geometric mean of precision and recall
    """
    precision = precision_score(y_true, y_proba, thres)
    recall = recall_score(y_true, y_proba, thres)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def roc_auc_score(y_true, y_proba):
    """
    ROC AUC metric
    """
    roc_auc = tf.metrics.auc(y_true, y_proba)[1]
    K.get_session().run(tf.local_variables_initializer())
    return roc_auc


"""
customized Keras layers for deep neural networks
"""


class Attention(Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: (samples, steps, features).
    # Output shape
        2D tensor with shape: (samples, features).
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True. # noqa
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Attention())
    """
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    """
    Keras Layer that implements a Capsule for temporal data.
    Literature publication: https://arxiv.org/abs/1710.09829v1
    Youtube video introduction: https://www.youtube.com/watch?v=pPN8d0E3900
    # Input shape
        4D tensor with shape: (samples, steps, features).
    # Output shape
        3D tensor with shape: (samples, num_capsule, dim_capsule).
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True. # noqa
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(
            LSTM(
                64,
                return_sequences=True, 
                recurrent_initializer=orthogonal(gain=1.0, seed=10000)
            )
        )
        model.add(
            Capsule(
                num_capsule=10,
                dim_capsule=10,
                routings=4,
                share_weights=True
            )
        )
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),   # noqa
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),   # noqa
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))    # noqa
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]  # noqa

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]  # noqa
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]    # noqa
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))    # noqa
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class DropConnect(Wrapper):
    """
    Keras Wrapper that implements a DropConnect Layer.
    When training with Dropout, a randomly selected subset of activations are
    set to zero within each layer. DropConnect instead sets a randomly
    selected subset of weights within the network to zero.
    Each unit thus receives input from a random subset of units in the
    previous layer.

    Reference: https://cs.nyu.edu/~wanli/dropc/
    Implementation: https://github.com/andry9454/KerasDropconnect
    """
    def __init__(self, layer, prob, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(
                K.dropout(self.layer.kernel, self.prob),
                self.layer.kernel)
            self.layer.bias = K.in_train_phase(
                K.dropout(self.layer.bias, self.prob),
                self.layer.bias)
        return self.layer.call(x)
