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
    re_tok = re.compile(r'[^\w\s]')
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
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def clean_numbers(text):
    """
    replace number with hash
    """
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def tokenize(text, remove_punct=False):
    """
    tokenize text into list of tokens
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
    # 5. clean number
    text = clean_numbers(text)
    # 6. tokenize
    tokens = text.split()
    return tokens


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


def pipeline_tokenize(text, remove_punct=False,
                      stemmer=True, lemmatizer=False):
    """
    1. clean text
    2. tokenize
    3. clean tokens
    """
    tokens = tokenize(text, remove_punct)
    tokens = clean_tokens(tokens, stemmer, lemmatizer)
    return tokens
