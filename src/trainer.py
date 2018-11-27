import os
import time
import argparse
# import pandas as pd
from eval import load_and_preprocess
from sklearn.model_selection import train_test_split


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
    print('Start to train model')
    model = model.train(X_train, y_train, X_val, y_val)
    print("Best iteration: {:.4f} with AUC ROC: {}".format(model.best_iteration, 'NA'))  # noqa
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Evaluation and Create Submission")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v1',
                        help='model version')
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

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load and preprocess data
    df_train, X_train = load_and_preprocess(datapath, module)
    # 3. train and eval
    X_t, X_v, y_t, y_v = train_test_split(
        X_train, df_train.target,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        shuffle=SHUFFLE, stratify=df_train.target)
    model = train_and_eval(X_t, y_t, X_v, y_v, module)
    # filepath = os.path.join(datapath, model + '.csv')
    # df_score.to_csv(filepath)
    # print('Save CV score file to {}'.format(filepath))
