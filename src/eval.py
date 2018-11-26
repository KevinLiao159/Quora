import os
import time
import operator
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics


def train_and_eval(X_train, y_train, X_val, y_val, module):
    """

    """


def load_and_preprocess(datapath, module):
    """
    load and preprocess

    Parameters
    ----------
    datapath: str, data directory that contains train.csv

    module: a python module

    Returns
    -------
    df_train: dataframe with raw text

    X_train: train data with proper features for model
    """
    t0 = time.time()
    print("Loading data")
    df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
    print("Train data with shape : ", df_train.shape)
    # transform
    X_train = module.transform(df_train['question_text'])
    print('Load and preprocessing took {:.2f}s'.format(time.time() - t0))
    return df_train, X_train


def fit_and_eval(X_train, y_train, X_val, y_val, module):
    """
    train model and eval hold-out performance
    BTW, write scores to csv files

    Parameters
    ----------
    X_train, y_train, X_val, y_val: features and targets

    module: a python module

    Return
    ------
    best_thres: float

    df_score: dataframe with thres and f1 score
    """
    # get model
    model = module.get_model()
    # train model
    print('Start to train model')
    model = model.fit(X_train, y_train)
    # predict probas
    print('Start to predict probas')
    y_proba = model.predict_proba(X_val)
    # score
    scores = {}
    for thres in np.arange(0, 0.51, 0.01):
        thres = round(thres, 2)
        scores[thres] = metrics.f1_score(y_val, (y_proba > thres).astype(int))
        print("Val F1 Score: {:.4f} with Threshold at {}".format(scores[thres], thres)) # noqa
    # get max
    best_thres, best_score = max(scores.items(), key=operator.itemgetter(1))
    print("Best F1 Score: {:.4f} with Threshold at {}".format(best_score, best_thres))  # noqa
    # write to disk
    df_score = pd.DataFrame(scores, index=['f1']).transpose()
    return best_thres, df_score


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Evaluation and Create Submission")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v0',
                        help='model version')
    parser.add_argument('--cv', type=int, default=2,
                        help='n folds for CV')
    return parser.parse_args()


if __name__ == '__main__':
    # config
    RANDOM_STATE = 99
    SHUFFLE = True
    TEST_SIZE = 0.20
    # get args
    args = parse_args()
    datapath = args.datapath
    model = args.model
    cv = args.cv

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load and preprocess data
    df_train, X_train = load_and_preprocess(datapath, module)
    # 3. train and eval
    # TODO;
    # 4. fit and eval
    if cv == 2:
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, df_train.target,
            test_size=TEST_SIZE, random_state=RANDOM_STATE,
            shuffle=SHUFFLE, stratify=df_train.target)
        best_thres, df_score = fit_and_eval(X_t, y_t, X_v, y_v, module)
        filepath = os.path.join(datapath, model + '.csv')
        df_score.to_csv(filepath)
        print('Save CV score file to {}'.format(filepath))
    else:
        cv_strat = StratifiedKFold(
            n_splits=cv, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        avg_thres = 0
        score_dfs = []
        for idx_train, idx_val in cv_strat.split(X_train, df_train.target):
            X_t = X_train[idx_train]
            y_t = df_train.target[idx_train]
            X_v = X_train[idx_val]
            y_v = df_train.target[idx_val]
            best_thres, df_score = fit_and_eval(X_t, y_t, X_v, y_v, module)
            avg_thres += best_thres
            score_dfs.append(df_score)
        best_thres = round(np.mean(avg_thres), 2)
        filepath = os.path.join(datapath, model + '.csv')
        pd.concat(score_dfs, axis=1).to_csv(filepath)
        print('Save CV score file to {}'.format(filepath))
    print('All done and it took {:.2f}s'.format(time.time() - t0))
