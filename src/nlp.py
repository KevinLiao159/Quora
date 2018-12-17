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
    text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)

    # general
    text = re.sub(r"(I|i)(\'|\’)m", "i am", text)
    text = re.sub(r"(A|a)in(\'|\’)t", "is not", text)
    text = re.sub(r"n(\'|\’)t", " not", text)
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
    regular_punct = list(string.punctuation)
    extra_punct = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&',
        '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
        '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
        '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
        '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
        '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
        'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
        '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
        '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
    all_punct = ''.join(sorted(list(set(regular_punct + extra_punct))))
    re_tok = re.compile(f'([{all_punct}])')
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
Extra - regex count-based features

    1. meta features
    2. topic features
        E.g. equality, politics, geopolitics, sexuallity,
            racism, sexism, ethnicity, movement, policy,
            corruption, religion
"""


def meta_features_transformer(df_text, col='question_text'):
    """
    transform dataframe with a text column to dataframe with
    following count based features
        1. number of words in the text
        2. number of unique words in the text
        3. number of characters in the text
        4. number of stopwords
        5. number of punctuations
        6. number of upper case words
        7. number of title case words
        8. average length of the words
    """
    # make sure it is a dataframe
    df_text = pd.DataFrame(df_text)
    # number of words in the text
    df_text["num_words"] = \
        df_text[col].apply(lambda x: len(str(x).split()))
    # number of unique words in the text
    df_text["num_unique_words"] = \
        df_text[col].apply(lambda x: len(set(str(x).split())))
    # number of characters in the text
    df_text["num_chars"] = \
        df_text[col].apply(lambda x: len(str(x)))
    # number of stopwords in the text
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
    df_text["num_stopwords"] = df_text[col].apply(lambda x: len(
            [w for w in str(x).lower().split() if w in stopwords]))
    # number of punctuations in the text
    df_text["num_punctuations"] = df_text[col].apply(lambda x: len(
        [c for c in str(x) if c in string.punctuation]))
    # number of title case words in the text
    df_text["num_words_upper"] = df_text[col].apply(lambda x: len(
        [w for w in str(x).split() if w.isupper()]))
    # number of title case words in the text
    df_text["num_words_title"] = df_text[col].apply(lambda x: len(
        [w for w in str(x).split() if w.istitle()]))
    # Average length of the words in the text
    df_text["mean_word_len"] = df_text[col].apply(lambda x: np.mean(
        [len(w) for w in str(x).split()]))
    return df_text[[c for c in df_text.columns if c != col]]


def regex_cnt(regex, text):
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


def topic_features_transformer(df_text, col='question_text'):
    """
    extract count of regex from text

    Parameters
    ----------
    df_text: dataframe

    Return
    ------
    dataframe with count features
    """
    key_word_regex = {
        # toxic
        'number_fuck': r"[Ff]\S{,3}[Kk]",
        'number_shit': r"[Ss][Hh]\S*[Tt]",
        'number_sick': r"[Ss][Ii]*[Cc]*[Kk]",
        'number_suck': r"[Ss][Uu]*[Cc]*[Kk]",
        'number_dick': r"[Dd]ick",
        'number_penis': r"[Pp]enis",
        'number_kill': r"[Kk]ill",
        'number_dead': r"[Dd]ea(d|th)",
        'number_ugly': r"ugly",
        'number_dumb': r"[Dd]umb",
        'number_idiot': r"[Ii]diot",
        'number_retards': r"[Rr]etard(s|)",
        'number_stupid': r"[S|s]tupid",
        'number_ass': r"\W[Aa][Ss]{2,}",
        'number_holes': r"[^[Ww][Hh]ole(s|)\W",
        'number_rape': r"\Wrap[ie][^d]",
        'number_anti': r"\W[Aa]nti",
        'number_hate': r"hate",
        'number_you': r"\W[Yy][Oo][Uu]\W",
        # racist
        'number_R': r"\W[Rr]acis[tm]",
        'number_color': r"[Cc]olor",
        'number_Muslim': r"[Mm]uslim",
        'number_terrorist': r"terror",
        'number_Islam': r"[Ii]s[Ll]am",
        'number_white': r"[Ww]hite",
        'number_India': r"[Ii]ndia",
        'number_Black': r"[Bb]lack",
        'number_Jew': r"[Jj]ew",
        'number_Hindu': r"[Hh]ind[ui]",
        'number_Asian': r"[Aa]sia",
        'number_phobi': r"phobi[ac]",
        'number_slaves': r"slave",
        # geo nation
        'number_Earth': r"Earth",
        'number_nation': r"[Nn]ation",
        'number_country': r"countr(y|ies)",
        'number_America': r"[Aa]merica",
        'number_United': r"United",
        'number_States': r"States",
        'number_USA': r"USA",
        'number_China': r"Chin(a|ese)",
        'number_Israel': r"[Ii]srael",
        'number_Pakistan': r"[Pp]akistan",
        'number_British': r"[Bb]rit(ish|ain)",
        'number_UK': r"UK",
        'number_Korea': r"[Kk]orea",
        'number_Russia': r"[Rr]ussia",
        'number_African': r"[Aa]frica",
        'number_Europe': r"[Ee]urope",
        'number_Japanese': r"[Jj]apan",
        'number_Palestinian': r"Palestinian",
        'number_Germany': r"German",
        'number_Arab': r"Arab",
        'number_Canada': r"Canada",
        'number_North': r"North",
        'number_South': r"South",
        'number_East': r"East",
        'number_West': r"West",
        'number_Middle': r"Middle",
        'number_Turks': r"Turks",
        'number_Kashmir': r"Kashmir",
        'number_Syria': r"Syria",
        'number_Australia': r"Australia",
        # politics
        'number_politic': r"politic",
        'number_President': r"[Pp]resident",
        'number_war': r"\W[Ww]ar(s|)\W",
        'number_Trump': r"[Tt]rump",
        'number_Donald': r"[Dd]onald",
        'number_Obama': r"[Oo]bama",
        'number_care': r"care\W",
        'number_Hillary': r"[Hh]illary",
        'number_Clinton': r"[Cc]linton",
        'number_Putin': r"Putin",
        'number_liberal': r"[Ll]iberal",
        'number_conservatives': r"conservative",
        'number_Democrats': r"Democrat",
        'number_Republicans': r"Republic",
        'number_Modi': r"Modi",
        'number_support': r"support",
        'number_media': r"media",
        'number_rights': r"right",
        'number_control': r"control",
        'number_claim': r"claim",
        'number_fake': r"fake",
        'number_poor': r"poor",
        'number_law': r"law",
        'number_legal': r"legal",
        'number_gun': r"\Wgun(s|)\W",
        'number_shooting': r"shoot",
        'number_immigrant': r"immigra",
        'number_citizens': r"citizen",
        'number_murder': r"murder",
        'number_deny': r"deny",
        'number_propaganda': r"propaganda",
        'number_refugee': r"refugee",
        'number_nuclear': r"nuclear",
        # sex
        'number_sex': r"[Ss]ex",
        'number_gender': r"gender",
        'number_love': r"love",
        'number_women': r"wom[ae]n",
        'number_men': r"m[ae]n",
        'number_girl': r"girl",
        'number_boy': r"boy",
        'number_female': r"\W[Ff]em[ai]",
        'number_male': r"\W[Mm]ale",
        'number_guy': r"guy",
        'number_gay': r"gay",
        'number_lesbian': r"lesbian",
        'number_trans': r"\W[Tt]rans",
        'number_son': r"\W[Ss]on\W",
        'number_sister': r"sister",
        'number_daughter': r"daughter",
        'number_marry': r"marr[yi]",
        'number_wife': r"wife",
        'number_mon': r"[Mm]on",
        'number_kid': r"\Wkid",
        'number_castrated': r"castrat",
        # religion
        'number_religion': r"religion",
        'number_God': r"God",
        'number_Jesus': r"Jesus",
        'number_Christianity': r"Christian",
        'number_atheists': r"athei",
        'number_Hitler': r"Hitler",
        'number_ISIS': r"ISIS",
        'number_Nazi': r"Nazi",
        # other
        'number_Quora': r"[Qq]uora",
        'number_genocide': r"genocide",
        'number_equal': r"\={2}.+\={2}",
        'number_quote': r"\"{4}\S+\"{4}"
    }
    # make sure it is a dataframe
    df_text = pd.DataFrame(df_text)
    # get features
    for col_name, regex in key_word_regex.items:
        df_text[col_name] = df_text[col].apply(lambda t: regex_cnt(regex, t))
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
