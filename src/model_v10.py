import nlp
from tqdm import tqdm


def preprocess(df_text, remove_punct=True,
               stemmer=True, lemmatizer=False):
    """
    preprocess and clean text from dataframe

    Parameters
    ----------
    df_text: dataframe, single column text

    remove_punct: bool

    stemmer: bool

    lemmatizer: bool

    Return
    ------
    df_preprocess
    """
    # progress bar
    tqdm.pandas()
    return df_text.progress_apply(lambda text: nlp.pipeline_tokenize(
        text, remove_punct, stemmer, lemmatizer))
