import os
import time
import argparse
import numpy as np
import pandas as pd

from utils import timer


def load_and_preprocess(datapath, module):
    """
    load and preprocess
    Parameters
    ----------
    datapath: str, data directory that contains train.csv and test.csv
    module: a python module
    Returns
    -------
    df_train, df_test: dataframe with raw text
    X_train, X_test: matrix with proper features
    """
    print("loading data ......")
    df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
    df_test = pd.read_csv(os.path.join(datapath, "test.csv"))
    train_test_cut = df_train.shape[0]
    print("train data with shape : ", df_train.shape)
    print("test data with shape : ", df_test.shape)
    # concat text data into single dataframe
    df_all = pd.concat(
        [df_train[['question_text']], df_test[['question_text']]],
        axis=0).reset_index(drop=True)
    # transform
    X_features = module.transform(df_all['question_text'])
    X_train = X_features[:train_test_cut]
    X_test = X_features[train_test_cut:]
    return df_train, df_test, X_train, X_test


def create_submission(X_train, y_train, X_test, df_test, thres, module):
    """
    train model with entire training data, predict test data,
    and create submission file

    Parameters
    ----------
    X_train, y_train, X_test: features and targets

    df_test: dataframe, test data

    thres: float, a decision threshold for classification

    module: a python module

    Return
    ------
    df_summission
    """
    # get model
    model = module.get_model()
    # train model
    print('fitting model')
    model = model.fit(X_train, y_train)
    # predict
    print('predicting probas')
    y_pred = np.squeeze(model.predict_proba(X_test) > thres).astype('int')
    # create submission file
    return pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred})


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Create Submission")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],   # noqa
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v0',
                        help='model version')
    parser.add_argument('--thres', type=float, default=0.23,
                        help='decision threshold for classification')
    return parser.parse_args()


if __name__ == '__main__':
    # config
    # SHUFFLE = True
    # get args
    args = parse_args()
    datapath = args.datapath
    model = args.model
    best_thres = args.thres

    t0 = time.time()
    # 1. import module
    module = __import__(model)
    # 2. load and preprocess data
    with timer("Load and Preprocess"):
        df_train, df_test, X_train, X_test = load_and_preprocess(datapath, module)  # noqa
    # 3. create submission file
    with timer('Trainning and Creating Submission'):
        filepath = os.path.join(datapath, 'submit_' + model + '.csv')
        df_submission = create_submission(
            X_train, df_train.target,
            X_test, df_test,
            best_thres, module)
        df_submission.to_csv(filepath, index=False)
        print('Save submission file to {}'.format(filepath))
    # record time spent
    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0)) # noqa
