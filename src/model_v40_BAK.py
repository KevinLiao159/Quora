"""
NN model with glove embeddings
layers:
    1. embedding layer (glove)
    2. SpatialDropout1D (0.1)
    3. bidirectional lstm & gru
    4. [global_max_pooling1d, attention, features]
    5. dense 64 & 32
    6. output (sigmoid)
"""
import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from keras.layers import (Input, Embedding, SpatialDropout1D, Bidirectional,
                          LSTM, GRU, GlobalMaxPool1D, Concatenate, Dropout,
                          Dense)
from keras.models import Model

from neural_networks import Attention
from neural_networks import NeuralNetworkClassifier

from tqdm import tqdm
tqdm.pandas()


# model configs
MAX_FEATURES = int(2.5e5)  # total word count = 227,538; clean word count = 186,551   # noqa
MAX_LEN = 80    # mean_len = 12; Q99_len = 40; max_len = 189;
RNN_UNITS = 40
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 16


# file configs
MODEL_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'models',
    'model_v50.hdf5'
)
EMBED_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'embeddings',
    'glove.840B.300d',
    'glove.pkl'
)


def get_network(embed_filepath):
    # features network
    input_features = Input(shape=(1, ), name='input_features')
    dense_features = Dense(units=16,
                           activation='relu',
                           name='dense_features')(input_features)
    # tokens network
    input_tokens = Input(shape=(MAX_LEN, ), name='input_tokens')
    # 1. embedding layer
    # get embedding weights
    print('load pre-trained embedding weights ......')
    embed_weights = pd.read_pickle(embed_filepath)
    input_dim = embed_weights.shape[0]
    output_dim = embed_weights.shape[1]
    x = Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        weights=[embed_weights],
        trainable=False,
        name='embedding'
    )(input_tokens)
    # clean up
    del embed_weights, input_dim, output_dim
    gc.collect()
    # 2. dropout
    x = SpatialDropout1D(rate=0.15)(x)
    # 3. bidirectional lstm & gru
    x = Bidirectional(
        layer=LSTM(RNN_UNITS, return_sequences=True),
        name='bidirectional_lstm'
    )(x)
    x = Bidirectional(
        layer=GRU(RNN_UNITS, return_sequences=True),
        name='bidirectional_gru'
    )(x)
    # 4. concat global_max_pooling1d and attention
    max_pool = GlobalMaxPool1D(name='global_max_pooling1d')(x)
    atten = Attention(step_dim=MAX_LEN, name='attention')(x)
    x = Concatenate(axis=-1)([max_pool, atten, dense_features])
    # 5. dense
    x = Dense(units=DENSE_UNITS_1, activation='relu', name='dense_1')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=DENSE_UNITS_2, activation='relu', name='dense_2')(x)
    # 6. output (sigmoid)
    output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    return Model(inputs=[input_tokens, input_features], outputs=output_layer)


def get_model():
    print('build network ......')
    model = get_network(embed_filepath=EMBED_FILEPATH)
    print(model.summary())
    return NeuralNetworkClassifier(
        model,
        balancing_class_weight=True,
        filepath=MODEL_FILEPATH)


"""
text cleaning
"""


def token_transformer(df_text):
    from model_v30 import transform
    return transform(df_text)


def features_transformer(df_text):
    from nlp import meta_features_transformer
    from nlp import topic_features_transformer
    # get features
    meta_features = meta_features_transformer(df_text).values
    topic_features = topic_features_transformer(df_text).values
    # concat
    joined_features = np.hstack[meta_features, topic_features]
    return minmax_scale(joined_features)


def transform(df_text):
    X1 = token_transformer(df_text)
    X2 = features_transformer(df_text)
    return [X1, X2]
