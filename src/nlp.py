import re
import string
import unicodedata
import nltk


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


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤$&#‘’])')
    return re_tok.sub(r' \1 ', text)


def remove_punctuation(text):
    """
    remove punctuation from text
    """
    re_tok = re.compile(f'([{string.punctuation}])')
    return re_tok.sub(' ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]{1,})')
    return re_tok.sub(r' \1 ', text)


def decontracted(text):
    """
    de-contract the contraction
    """
    # specific
    text = re.sub(r"(W|w)on\'t", "will not", text)
    text = re.sub(r"(C|c)an\'t", "can not", text)

    # general
    text = re.sub(r"(I|i)\'m", "i am", text)
    text = re.sub(r"(A|a)in\'t", "is not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    return text


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
    # 2. to lower
    text = text.lower()
    # 3. space
    text = spacing_punctuation(text)
    text = spacing_number(text)
    # (optional)
    if remove_punct:
        text = remove_punctuation(text)
    # 4. de-contract
    text = decontracted(text)
    # 5. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = clean_number(text)
    # 6. remove space
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
