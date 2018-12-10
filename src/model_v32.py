"""
NN model with fasttext embeddings
layers:
    1. embedding layer (fasttext)
    2. SpatialDropout1D (0.2)
    3. bidirectional lstm & gru
    4. global_max_pooling1d
    5. dense 32 & 16
    6. output (sigmoid)
"""
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from neural_networks import NeuralNetworkClassifier

from tqdm import tqdm
tqdm.pandas()


# model configs
MAX_FEATURES = int(2.5e5)  # total word count = 227,538; clean word count = 186,551   # noqa
MAX_LEN = 80    # mean_len = 12; Q99_len = 40; max_len = 189;


# file configs
MODEL_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'models',
    'model_v32.hdf5'
)
EMBED_FILEPATH = os.path.join(
    os.environ['DATA_PATH'],
    'embeddings',
    'wiki-news-300d-1M',
    'fasttext.pkl'
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


def preprocess(text):
    """
    preprocess text into clean text for tokenization

    NOTE:
        1. fasttext supports uppper case words
        2. fasttext supports digit
        3. fasttext supports punctuation
        4. fasttext supports domains e.g. www.apple.com
        5. fasttext supports madeup word e.g. fuckall
            (unclear it supports misspelled words)
    """
    from model_v30 import preprocess as preprocess_base
    text = preprocess_base(text)
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
