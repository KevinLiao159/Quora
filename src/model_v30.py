"""
NN model with glove embeddings
layers:
    1. embedding layer (glove)
    2. SpatialDropout1D (0.2)
    3. bidirectional lstm & gru
    4. global_max_pooling1d
    5. dense 32 & 16
    6. output (sigmoid)
"""
import os
import gc
import re
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


def get_network(embed_filepath):
    input_layer = Input(shape=(MAX_LEN, ), name='input')
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
    )(input_layer)
    # clean up
    del embed_weights, input_dim, output_dim
    gc.collect()
    # 2. dropout
    x = SpatialDropout1D(rate=0.2)(x)
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
    model = get_network(embed_filepath=EMBED_FILEPATH)
    print(model.summary())
    return NeuralNetworkClassifier(
        model,
        balancing_class_weight=True,
        filepath=MODEL_FILEPATH)


"""
text cleaning
"""


def clean_misspell(text):
    """
    misspell list (quora vs. glove)
    """
    misspell_to_sub = {
        '(T|t)erroristan': 'terrorist Pakistan',
        'BIMARU': 'Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh',
        '(H|h)induphobic': 'Hindu phobic',
        '(H|h)induphobia': 'Hindu phobic',
        'Babchenko': 'Arkady Arkadyevich Babchenko faked death',
        'Boshniaks': 'Bosniaks',
        'Dravidanadu': 'Dravida Nadu',
        'mysoginists': 'misogynists',
        'MGTOWS': 'Men Going Their Own Way',
        'mongloid': 'Mongoloid',
        'unsincere': 'insincere',
        'meninism': 'male feminism',
        'jewplicate': 'jewish replicate',
        'unoin': 'Union',
        'daesh': 'Islamic State of Iraq and the Levant',
        'Kalergi': 'Coudenhove-Kalergi',
        ' apist': ' Ape',
        '(B|b)hakts': 'Bhakt',
        'Tambrahms': 'Tamil Brahmin',
        'Pahul': 'Amrit Sanskar',
        'SJW(s|)': 'social justice warrior',
        'incel(s|)': 'involuntary celibates',
        'emiratis': 'Emiratis',
        'weatern': 'western',
        'Pizzagate': 'Pizzagate conspiracy theory',
        'naïve': 'naive',
        'Skripal': 'Sergei Skripal',
        '(R|r)emainers': 'remainer',
        'antibrahmin': 'anti Brahminism',
        'HYPSM': ' Harvard, Yale, Princeton, Stanford, MIT',
        'HYPS': ' Harvard, Yale, Princeton, Stanford',
        'kompromat': 'compromising material',
        '(T|t)harki': 'pervert',
        'mastuburate': 'masturbate',
        'Zoë': 'Zoe',
        'indans': 'Indian',
        'xender': 'gender',
        'Naxali': 'Naxalite',
        'Bathla': 'Namit Bathla',
        'Mewani': 'Indian politician Jignesh Mevani',
        'clichéd': 'cliché',
        'cliché(s|)': 'cliché',
        'Wjy': 'Why',
        'Fadnavis': 'Indian politician Devendra Fadnavis',
        'Awadesh': 'Indian engineer Awdhesh Singh',
        'Awdhesh': 'Indian engineer Awdhesh Singh',
        'Khalistanis': 'Sikh separatist movement',
        'madheshi': 'Madheshi',
        'Quorans': 'Quoran',
        'BNBR': 'Be Nice, Be Respectful',
        'Bolsonaro': 'Jair Bolsonaro',
        'XXXTentacion': 'Tentacion',
        'Padmavat': 'Indian Movie Padmaavat',
        'Žižek': 'Slovenian philosopher Slavoj Žižek',
        'Adityanath': 'Indian monk Yogi Adityanath',
        '(B|b)rexit': 'British Exit',
        'jallikattu': 'Jallikattu',
        'fortnite': 'Fortnite',
        'Swachh': 'Swachh Bharat mission campaign',
        'Qoura': 'Quora',
        'narcissit': 'narcissist',
        # extra in sample
        'Doklam': 'Tibet',
        'Drumpf': 'Donald Trump',
        'Strzok': 'Hillary Clinton scandal',
        'rohingya': 'Rohingya',
        'wumao': 'offensive Chinese',
        'Sanghis': 'Sanghi',
        'Tamilans': 'Tamils',
        'biharis': 'Biharis',
        'Rejuvalex': 'hair growth formula',
        'Feku': 'The Man of India',
        'deplorables': 'deplorable',
        'muhajirs': 'Muslim immigrants',
        'Brexiters': 'British Exit supporters',
        'Brexiteers': 'British Exit supporters',
        'Brexiting': 'British Exit',
        'Gujratis': 'Gujarati',
        'Chutiya': 'Tibet people',
        'thighing': 'masturbate',
        '卐': 'Nazi Germany',
        'rohingyas': 'Muslim ethnic group',
        'Pribumi': 'Native Indonesians',
        'Gurmehar': 'Gurmehar Kaur Indian student activist',
        'Novichok': 'Soviet Union agents',
        'Khazari': 'Khazars',
        'Demonetization': 'demonetization',
        'demonetisation': 'demonetization',
        'cryptocurrencies': 'bitcoin',
        'Hindians': 'offensive Indian',
        'vaxxers': 'vocal nationalists',
        'remoaners': 'remainer',
        'Jewism': 'Judaism',
        'Eroupian': 'European',
        'WMAF': 'White male Asian female',
        'moeslim': 'Muslim',
        'cishet': 'cisgender and heterosexual person',
        'Eurocentrics': 'Eurocentrism',
        'Jewdar': 'Jew dar',
        'Asifas': 'abduction, rape, murder case',
        'marathis': 'Marathi',
        'Trumpanzees': 'Trump chimpanzee',
        'quoras': 'Quora',
        'Crimeans': 'Crimea people',
        'atrracted': 'attract',
        'LGBT': 'lesbian, gay, bisexual, transgender',
        'Boshniaks': 'Bosniaks',
        'Myeshia': 'widow of Green Beret killed in Niger',
        'demcoratic': 'Democratic',
        'raaping': 'rape',
        'Dönmeh': 'Islam',
        'feminazism': 'feminism nazi',
        'Quroa': 'Quora',
        'QUORA': 'Quora',
        'langague': 'language',
        '(H|h)ongkongese': 'HongKong people',
        '(K|k)ashmirians': 'Kashmirian',
        '(C|c)hodu': 'fucker',
        'penish': 'penis',
        'micropenis': 'small penis',
        'Madridiots': 'Madrid idiot',
        'Ambedkarites': 'Dalit Buddhist movement',
        'ReleaseTheMemo': 'cry for the right and Trump supporters',
        'harrase': 'harass',
        '(B|b)arracoon': 'Black slave',
        '(C|c)astrater': 'castration',
        '(R|r)apistan': 'rapist Pakistan',
        '(T|t)urkified': 'Turkification',
        'Dumbassistan': 'dumb ass Pakistan',
        'facetards': 'Facebook retards',
        'rapefugees': 'rapist refugee',
        'superficious': 'superficial'
    }
    for mis, sub in misspell_to_sub.items():
        text = re.sub(mis, sub, text)
    return text


def spacing_misspell(text):
    """
    'deadbody' -> 'dead body'
    """
    misspell_list = [
        'body',
        '(D|d)ead',
        '(N|n)orth',
        '(K|k)orea',
        'matrix',
        '(S|s)hit',
        '(F|f)uck',
        '(F|f)uk',
        '(F|f)ck',
        '(D|d)ick',
        'Trump',
        '\W(A|a)nti',
        '(W|w)hy',
        # 'Jew',
        'bait',
        'care',
        'troll',
        'over',
        'gender',
        'people',
        'kind',
        '(S|s)ick',
        '(S|s)uck',
        '(I|i)diot',
        # 'hole(s|)\W',
        '(B|b)ooty',
        '(C|c)oin(s|)\W',
        '\W(N|n)igger'
    ]
    for word in misspell_list:
        text = re.sub(r"({})".format(word), r" \1 ", text)
    return text


def clean_latex(text):
    """
    convert r"[math]\vec{x} + \vec{y}" to English
    """
    pattern_to_sub = {
        r'\[math\]': ' LaTex math ',
        r'\[\/math\]': ' LaTex math ',
        r'\\mathrm': ' LaTex math mode ',
        r'\\mathbb': ' LaTex math mode ',
        r'\\boxed': ' LaTex equation ',
        r'\\begin': ' LaTex equation ',
        r'\\end': ' LaTex equation ',
        r'\\left': ' LaTex equation ',
        r'\\right': ' LaTex equation ',
        r'\\(over|under)brace': ' LaTex equation ',
        r'\\text': ' LaTex equation ',
        r'\\vec': ' vector ',
        r'\\var': ' variable ',
        r'\\theta': ' theta ',
        r'\\mu': ' average ',
        r'\\min': ' minimum ',
        r'\\max': ' maximum ',
        r'\\sum': ' + ',
        r'\\times': ' * ',
        r'\\cdot': ' * ',
        r'\\hat': ' ^ ',
        r'\\frac': ' / ',
        r'\\div': ' / ',
        r'\\sin': ' Sine ',
        r'\\cos': ' Cosine ',
        r'\\tan': ' Tangent ',
        r'\\infty': ' infinity ',
        r'\\int': ' integer ',
        r'\\in': ' in ',
        r'\\': ' LaTex ',
    }
    for pat, sub in pattern_to_sub.items():
        text = re.sub(pat, sub, text)
    return text


def preprocess(text, remove_num=False):
    """
    preprocess text into clean text for tokenization

    NOTE:
        1. glove supports uppper case words
        2. glove supports digit
        3. glove supports punctuation
        5. glove supports domains e.g. www.apple.com
        6. glove supports misspelled words e.g. FUCKKK
    """
    # 1. normalize
    text = normalize_unicode(text)
    # 2. remove new line
    text = remove_newline(text)
    # 3. de-contract
    text = decontracted(text)
    # 4. clean misspell
    text = clean_misspell(text)
    # 5. space misspell
    text = spacing_misspell(text)
    # 6. clean_latex
    text = clean_latex(text)
    # 7. space
    text = spacing_punctuation(text)
    # 8. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = spacing_digit(text)
    # 9. remove space
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
