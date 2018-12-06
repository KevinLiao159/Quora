"""
NN model with glove embeddings
layers:
    1. embedding layer (glove)
    2. SpatialDropout1D (0.1)
    3. bidirectional lstm & gru
    4. global_max_pooling1d
    5. dense 32 & 16
    6. output (sigmoid)
"""
import os
import gc
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Input, Embedding, SpatialDropout1D, Bidirectional,
                          LSTM, GRU, GlobalMaxPool1D, Dense)
from keras.models import Model

from neural_networks import NeuralNetworkClassifier
from nlp import (normalize_unicode, remove_newline, decontracted,
                 spacing_punctuation, remove_number, spacing_digit,
                 remove_space)

from tqdm import tqdm
tqdm.pandas()


# # toy configs
# MAX_FEATURES = int(5e3)
# MAX_LEN = 20
# RNN_UNITS = 16
# DENSE_UNITS_1 = 8
# DENSE_UNITS_2 = 4

# model configs
MAX_FEATURES = int(2.5e5)  # total word count = 227,538; clean word count = 186,551   # noqa
MAX_LEN = 80    # mean_len = 12; Q99_len = 40; max_len = 189;
RNN_UNITS = 40
DENSE_UNITS_1 = 32
DENSE_UNITS_2 = 16


# file configs
MODEL_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'models',
    'model_v30.hdf5'
)
EMBED_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'embeddings',
    'glove.840B.300d',
    'glove.pkl'
)


def get_network():
    input_layer = Input(shape=(MAX_LEN, ), name='input')
    # 1. embedding layer
    # get embedding weights
    print('load pre-trained embedding weights ......')
    embed_weights = pd.read_pickle(EMBED_FILEPATH)
    input_dim = embed_weights.shape[0]
    output_dim = embed_weights.shape[1]
    x = Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        weights=[embed_weights],
        trainable=False,
        name='embedding_glove'
    )(input_layer)
    # clean up
    del embed_weights, input_dim, output_dim
    gc.collect()
    # 2. dropout
    x = SpatialDropout1D(rate=0.1)(x)
    # 3. bidirectional lstm & gru
    x = Bidirectional(
        layer=LSTM(RNN_UNITS, return_sequences=True),
        name='bidirectional_lstm'
    )(x)
    x = Bidirectional(
        layer=GRU(RNN_UNITS, return_sequences=True),
        name='bidirectional_gru'
    )(x)
    # 4. global_max_pooling1d
    x = GlobalMaxPool1D(name='global_max_pooling1d')(x)
    # 5. dense
    x = Dense(units=DENSE_UNITS_1, activation='relu', name='dense_1')(x)
    x = Dense(units=DENSE_UNITS_2, activation='relu', name='dense_2')(x)
    # 6. output (sigmoid)
    output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    return Model(inputs=input_layer, outputs=output_layer)


def get_model():
    print('build network ......')
    model = get_network()
    print(model.summary())
    return NeuralNetworkClassifier(
        model,
        balancing_class_weight=True,
        filepath=MODEL_FILEPATH)


"""
text cleaning
"""


def preprocess(text, remove_num=True):
    """
    preprocess text into clean text for tokenization
    """
    # 1. normalize
    text = normalize_unicode(text)
    # 2. remove new line
    text = remove_newline(text)
    # 3. de-contract
    text = decontracted(text)
    # 4. space
    text = spacing_punctuation(text)
    # 5. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = spacing_digit(text)
    # 6. remove space
    text = remove_space(text)
    return text


def tokenize(df_text):
    # preprocess
    df_text = df_text.progress_apply(preprocess)
    # tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_FEATURES,
        filters='',
        lower=False,
        split=' ')
    # fit to data
    tokenizer.fit_on_texts(list(df_text))
    # tokenize the texts into sequences
    sequences = tokenizer.texts_to_sequences(df_text)
    return sequences, tokenizer


def transform(df_text):
    seqs, _ = tokenize(df_text)
    # pad the sentences
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='pre', truncating='post')
    return X
