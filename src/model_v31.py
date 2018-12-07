"""
NN model with paragram embeddings
layers:
    1. embedding layer (paragram)
    2. SpatialDropout1D (0.1)
    3. bidirectional lstm & gru
    4. global_max_pooling1d
    5. dense 32 & 16
    6. output (sigmoid)
"""
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nlp import (normalize_unicode, remove_newline, decontracted,
                 spacing_punctuation, remove_number, spacing_digit,
                 remove_space)
from neural_networks import NeuralNetworkClassifier


# model configs
MAX_FEATURES = int(2.5e5)  # total word count = 227,538; clean word count = 186,551   # noqa
MAX_LEN = 80    # mean_len = 12; Q99_len = 40; max_len = 189;


# file configs
MODEL_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'models',
    'model_v31.hdf5'
)
EMBED_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'embeddings',
    'paragram_300_sl999',
    'paragram.pkl'
)


def get_model():
    from model_v30 import get_network
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


def preprocess(text, remove_num=True):
    """
    preprocess text into clean text for tokenization

    NOTE:
        1. paragram does NOT support uppper case words
            (all lower case)
        2. paragram supports digit
        3. paragram supports punctuation
        4. glove supports domains e.g. www.apple.com
        5. paragram supports madeup word e.g. whathefuck
            (unclear it supports misspelled words)
    """
    # 1. normalize
    text = normalize_unicode(text)
    # 2. remove new line
    text = remove_newline(text)
    # 3. to lower
    text = text.lower()
    # 4. de-contract
    text = decontracted(text)
    # 5. space
    text = spacing_punctuation(text)
    # 6. handle number
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
