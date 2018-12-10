"""
Build embedding weights for Keras embedding layer
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from utils import timer


"""
pre-trained embeddeding vector loader
"""


def load_word_embedding(filepath):
    """
    given a filepath to embeddings file, return a word to vec
    dictionary, in other words, word_embedding

    E.g. {'word': array([0.1, 0.2, ...])}
    """
    def _get_vec(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    print('load word embedding ......')
    try:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(filepath))
    except UnicodeDecodeError:
        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(
            filepath, encoding="utf8", errors='ignore'))
    # sanity check word vector length
    words_to_del = []
    for word, vec in word_embedding.items():
        if len(vec) != 300:
            words_to_del.append(word)
    for word in words_to_del:
        del word_embedding[word]
    return word_embedding


def create_embedding_weights(word_index, word_embedding, max_features):
    """
    create weights for embedding layer where row is the word index
    and collumns are the embedding dense vector

    Parameters
    ----------
    word_index: dict, mapping of word to word index. E.g. {'the': 2}
        you can get word_index by keras.tokenizer.word_index

    word_embedding: dict, mapping of word to word embedding
        E.g. {'the': array([0.1, 0.2, ...])}
        you can get word_index by above function load_word_embedding and
        embedding filepath

    max_features: int, number of words that we want to keep

    Return
    ------
    embedding weights: np.array, with shape (number of words, 300)
    """
    print('create word embedding weights ......')
    # get entire embedding matrix
    mat_embedding = np.stack(word_embedding.values())
    # get shape
    a, b = min(max_features, len(word_index)), mat_embedding.shape[1]
    print('embedding weights matrix with shape: ({}, {})'.format(a, b))
    # init embedding weight matrix
    embedding_mean, embedding_std = mat_embedding.mean(), mat_embedding.std()
    embedding_weights = np.random.normal(embedding_mean, embedding_std, (a, b))
    # mapping
    for word, idx in word_index.items():
        if idx >= a:
            continue
        word_vec = word_embedding.get(word, None)
        if word_vec is not None:
            embedding_weights[idx] = word_vec
    return embedding_weights


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Build Embedding Weights For Keras Model")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--embedding', nargs='?', default='glove',
                        help='choose word embedding')
    parser.add_argument('--model', nargs='?', default='model_v30',
                        help='model version')
    return parser.parse_args()


if __name__ == '__main__':
    # mapper
    filepath_mapper = {
        'word2vec': os.path.join(
            os.environ['DATA_PATH'],
            'embeddings',
            'GoogleNews-vectors-negative300',
            'GoogleNews-vectors-negative300.bin'),
        'glove': os.path.join(
            os.environ['DATA_PATH'],
            'embeddings',
            'glove.840B.300d',
            'glove.840B.300d.txt'),
        'paragram': os.path.join(
            os.environ['DATA_PATH'],
            'embeddings',
            'paragram_300_sl999',
            'paragram_300_sl999.txt'),
        'fasttext': os.path.join(
            os.environ['DATA_PATH'],
            'embeddings',
            'wiki-news-300d-1M',
            'wiki-news-300d-1M.vec'),
    }
    # get args
    args = parse_args()
    datapath = args.datapath
    embedding = args.embedding
    model = args.model
    # get embedding filepath
    embed_filepath = filepath_mapper[embedding]

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load data and get word index
    with timer("Extract Word Index From Train and Test Data"):
        print("loading data ......")
        df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
        df_test = pd.read_csv(os.path.join(datapath, "test.csv"))
        print("train data with shape : ", df_train.shape)
        print("test data with shape : ", df_test.shape)
        # concat text data into single dataframe
        df_all = pd.concat(
            [df_train[['question_text']], df_test[['question_text']]],
            axis=0).reset_index(drop=True)
        # get word index
        print('tokenizing text ......')
        _, tokenizer = module.tokenize(df_all['question_text'])
        word_index = tokenizer.word_index
    # 3. create embedding weights matrix
    with timer("Create Embedding Weights Matrix"):
        # load word embeddings
        print('loading embedding file')
        word_embed = load_word_embedding(embed_filepath)
        # create embedding weights matrix
        print('create embedding weights ......')
        embed_weights = create_embedding_weights(
            word_index,
            word_embed,
            module.MAX_FEATURES)
        # pickle numpy file
        filepath_to_save = os.path.join(
            embed_filepath.rsplit('/', 1)[0],
            '{}.pkl'.format(embedding))
        pd.to_pickle(embed_weights, filepath_to_save)
        print('save embedding weights to {}'.format(filepath_to_save))
    # record time spent
    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0)) # noqa
