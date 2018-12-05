import os
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import timer, load_and_preprocess


def train_and_eval(X_train, y_train, X_val, y_val, module):
    """
    train model and eval hold-out performance
    BTW, write scores to csv files

    Parameters
    ----------
    X_train, y_train, X_val, y_val: features and targets

    module: a python module

    Return
    ------
    training logs
    """
    # get model
    model = module.get_model()
    # train model
    print('trainning model ......')
    model = model.train(X_train, y_train, X_val, y_val)
    best_param = model.best_param
    best_score = model.best_score
    print("best param: {:.4f} with best score: {}".format(best_param, best_score))  # noqa
    return pd.DataFrame({'best_param': [best_param], 'best_score': [best_score]})   # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Validation and Pick the Best Best Param")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v1',
                        help='model version')
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

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load and preprocess data
    with timer("Load and Preprocess"):
        df_train, _, X_train, _ = load_and_preprocess(datapath, module)
    # 3. train and eval
    with timer('Trainning and Tuning'):
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, df_train.target,
            test_size=TEST_SIZE, random_state=RANDOM_STATE,
            shuffle=SHUFFLE, stratify=df_train.target)
        df_score = train_and_eval(X_t, y_t, X_v, y_v, module)
        filepath = os.path.join(datapath, 'trainer_{}.csv'.format(model))
        df_score.to_csv(filepath)
        print('Save CV score file to {}'.format(filepath))
    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0)) # noqa
