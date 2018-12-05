import re
import string
import unicodedata
import nltk
import numpy as np
import pandas as pd

"""
text cleaning
"""


def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


def decontracted(text):
    """
    de-contract the contraction
    """
    # specific
    text = re.sub(r"(W|w)on(\'|\’)t", "will not", text)
    text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)
    text = re.sub(r"y(\'|\’)all", "you all", text)
    text = re.sub(r"Y(\'|\’)(A|a)ll", "You All", text)

    # general
    text = re.sub(r"i(\'|\’)m", "i am", text)
    text = re.sub(r"I(\'|\’)m", "I am", text)
    text = re.sub(r"I(\'|\’)M", "I Am", text)
    text = re.sub(r"ain(\'|\’)t", "is not", text)
    text = re.sub(r"Ain(\'|\’)t", "Is Not", text)
    text = re.sub(r"n(\'|\’)t", " not", text)
    text = re.sub(r"N(\'|\’)t", " Not", text)
    text = re.sub(r"(\'|\’)re", " are", text)
    text = re.sub(r"(\'|\’)s", " is", text)
    text = re.sub(r"(\'|\’)d", " would", text)
    text = re.sub(r"(\'|\’)ll", " will", text)
    text = re.sub(r"(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)ve", " have", text)

    return text


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤$&#‘’''])')
    return re_tok.sub(r' \1 ', text)


def remove_punctuation(text):
    """
    remove punctuation from text
    """
    re_tok = re.compile(f'([{string.punctuation}])')
    return re_tok.sub(' ', text)


def spacing_digit(text):
    """
    add space before and after digits
    """
    re_tok = re.compile('([0-9])')
    return re_tok.sub(r' \1 ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]{1,})')
    return re_tok.sub(r' \1 ', text)


def clean_number(text):
    """
    replace number with hash
    """
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text


def preprocess(text, remove_punct=False, remove_num=True):
    """
    preprocess text into clean text for tokenization
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
    text = spacing_number(text)
    # (optional)
    if remove_punct:
        text = remove_punctuation(text)
    # 6. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = clean_number(text)
    # 7. remove space
    text = remove_space(text)
    return text


def word_tokenize(text, remove_punct=False, remove_num=True):
    """
    tokenize text into list of word tokens
    """
    # 1. preprocess
    text = preprocess(text, remove_punct, remove_num)
    # 2. tokenize
    tokens = text.split()
    return tokens


def char_tokenize(text, remove_punct=False, remove_num=True):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = word_tokenize(text, remove_punct, remove_num)
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


"""
token cleaning
"""


def strip_space(tokens):
    """
    strip spaces
    """
    return [t.strip() for t in tokens]


def remove_stopwords(tokens):
    """
    remove stopwords from tokens
    """
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
    return [t for t in tokens if t.lower() not in stopwords]


def stemming(tokens):
    """
    stem tokens
    """
    porter = nltk.PorterStemmer()
    return [porter.stem(t) for t in tokens]


def lemmatize(tokens):
    """
    lemmatize tokens
    """
    try:
        wnl = nltk.WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')
        wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in tokens]


def clean_tokens(tokens, stemmer=True, lemmatizer=False):
    """
    cleaning
    """
    tokens = strip_space(tokens)
    tokens = remove_stopwords(tokens)
    if stemmer:
        tokens = stemming(tokens)
    if lemmatizer:
        tokens = lemmatize(tokens)
    return tokens


"""
NLP pipeline
"""


def word_analyzer(text, remove_punct=False, remove_num=True,
                  stemmer=True, lemmatizer=False):
    """
    1. clean text
    2. tokenize
    3. clean tokens
    """
    tokens = word_tokenize(text, remove_punct, remove_num)
    tokens = clean_tokens(tokens, stemmer, lemmatizer)
    return tokens


"""
Extra - other count-based features
"""


def count_regexp_occ(regex, text):
    """
    simple way to get the number of occurence of a regex

    Parameters
    ----------
    regex: str, regex of an interested pattern

    text: str

    Return
    ------
    count of occurence: int
    """
    return len(re.findall(regex, text))


def count_feature_transformer(df_text, col='question_text'):
    """
    extract count of regex from text

    Parameters
    ----------
    df_text: dataframe

    Return
    ------
    dataframe with count features
    """
    # make sure it is a dataframe
    df_text = pd.DataFrame(df_text)
    # Count number of \n
    df_text["ant_slash_n"] = df_text[col].apply(lambda x: count_regexp_occ(r"\n", x))   # noqa
    # Get length in words and characters
    df_text["raw_word_len"] = df_text[col].apply(lambda x: len(x.split()))
    df_text["raw_char_len"] = df_text[col].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df_text["nb_upper"] = df_text[col].apply(lambda x: count_regexp_occ(r"[A-Z]", x))   # noqa
    # Number of F words - f..k contains folk, fork,
    df_text["nb_fk"] = df_text[col].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))  # noqa
    # Number of S word
    df_text["nb_sk"] = df_text[col].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))  # noqa
    # Number of D words
    df_text["nb_dk"] = df_text[col].apply(lambda x: count_regexp_occ(r"[dD]ick", x))    # noqa
    # Number of occurence of You, insulting someone usually needs someone called : you  # noqa
    df_text["nb_you"] = df_text[col].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))    # noqa
    # Just to check you really refered to my mother ;-)
    df_text["nb_mother"] = df_text[col].apply(lambda x: count_regexp_occ(r"\Wmother\W", x)) # noqa
    # Just checking for toxic 19th century vocabulary
    df_text["nb_ng"] = df_text[col].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x)) # noqa
    # Some Sentences start with a <:> so it may help
    df_text["start_with_columns"] = df_text[col].apply(lambda x: count_regexp_occ(r"^\:+", x))  # noqa
    # Check for time stamp
    df_text["has_timestamp"] = df_text[col].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))   # noqa
    # Check for dates 18:44, 8 December 2010
    df_text["has_date_long"] = df_text[col].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))   # noqa
    # Check for date short 8 December 2010
    df_text["has_date_short"] = df_text[col].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))   # noqa
    # Check for http links
    df_text["has_http"] = df_text[col].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))  # noqa
    # check for mail
    df_text["has_mail"] = df_text[col].apply(lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x))  # noqa
    # Looking for words surrounded by == word == or """" word """"
    df_text["has_emphasize_equal"] = df_text[col].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))   # noqa
    df_text["has_emphasize_quotes"] = df_text[col].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x)) # noqa
    return df_text[[c for c in df_text.columns if c != col]]


"""
Extra - for keras

1. fast text
    link: https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py # noqa
"""


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}    # noqa
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def append_ngram(X, ngram=2):
    """
    Append ngram to sequences of input list.
    This is a step between Keras.tokenize and Keras.pad_sequences

    Parameters
    ----------
    X: list of list of token indices

    ngram: int

    Return
    ------
    X: list of list of original tokens and ngram tokens
    """
    print('adding {}-gram features ......'.format(ngram))
    # iterate seq to find num features and build ngram set
    num_features = []
    ngram_set = set()
    for input_list in X:
        # get max features to avoid collision with existing features
        num_features.append(max(input_list))
        # create set of unique n-gram from the training set
        for i in range(2, ngram + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
    # get max features from original tokens
    max_features = max(num_features)
    # map ngram to int and avoid collision with existing features
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    # get new max features after appending ngram features
    new_max_features = np.max(list(indice_token.keys())) + 1
    print('there is {} features in total after '
          'adding ngram'.format(new_max_features))
    # augmenting data with n-grams features
    X = add_ngram(X, token_indice, ngram)
    return X
