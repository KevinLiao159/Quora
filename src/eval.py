import os
import time
import operator
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

from utils import timer, load_and_preprocess, load_trained_model


def fit_and_eval(X_train, y_train, X_val, y_val, module, pretrained=False):
    """
    train model and eval hold-out performance

    BTW, write scores to csv files

    Parameters
    ----------
    X_train, y_train, X_val, y_val: features and targets

    module: a python module

    pretrained: bool, if true, load the model pickle

    Return
    ------
    best_thres: float

    df_score: dataframe with thres and f1 score
    """
    # get model
    model = module.get_model()
    # load model
    if pretrained:
        print('loading model ......')
        network = model.model
        model.model = load_trained_model(network, module.MODEL_FILEPATH)
    else:   # or, train model
        print('fitting model ......')
        model = model.fit(X_train, y_train)
    # predict probas
    print('predicting probas ......')
    y_proba = model.predict_proba(X_val)
    # score
    scores = {}
    for thres in np.arange(0, 0.51, 0.01):
        thres = round(thres, 2)
        scores[thres] = metrics.f1_score(y_val, (y_proba > thres).astype(int))
        print("val F1 score: {:.4f} with threshold at {}".format(scores[thres], thres)) # noqa
    # get max
    best_thres, best_score = max(scores.items(), key=operator.itemgetter(1))
    print("best F1 score: {:.4f} with threshold at {}".format(best_score, best_thres))  # noqa
    # write to disk
    df_score = pd.DataFrame(scores, index=['f1']).transpose()
    return best_thres, df_score


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Evaluation and Pick the Best Threshold")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v0',
                        help='model version')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='use pre-trained model')
    parser.add_argument('--cv', type=int, default=2,
                        help='n folds for CV')

    return parser.parse_args()


if __name__ == '__main__':
    # config
    RANDOM_STATE = 99
    SHUFFLE = True
    TEST_SIZE = 0.50
    # get args
    args = parse_args()
    datapath = args.datapath
    model = args.model
    pretrained = args.pretrained
    cv = args.cv

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load and preprocess data
    with timer("Load and Preprocess"):
        df_train, _, X_train, _ = load_and_preprocess(datapath, module)
    # 3. fit and eval
    with timer('Fitting and Validating'):
        if cv == 2:
            X_t, X_v, y_t, y_v = train_test_split(
                X_train, df_train.target,
                test_size=TEST_SIZE, random_state=RANDOM_STATE,
                shuffle=SHUFFLE, stratify=df_train.target)
            best_thres, df_score = fit_and_eval(X_t, y_t, X_v, y_v, module, pretrained) # noqa
            filepath = os.path.join(datapath, 'eval_{}.csv'.format(model))
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
            filepath = os.path.join(datapath, 'trainer_{}.csv'.format(model))
            pd.concat(score_dfs, axis=1).to_csv(filepath)
            print('Save CV score file to {}'.format(filepath))

    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0)) # noqa
