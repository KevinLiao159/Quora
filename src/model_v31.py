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
from neural_networks import NeuralNetworkClassifier


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
    'paragram_300_sl999.txt'
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


def tokenize(df_text):
    from model_v30 import tokenize
    return tokenize(df_text)


def transform(df_text):
    from model_v30 import transform
    return transform(df_text)
