import os
import time
import argparse
import pandas as pd


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

    train: train data with proper features for model

    test: test data with proper features for model
    """
    t0 = time.time()
    print("Loading data")
    df_train = pd.read_csv(os.path.join(datapath, "train.csv"))
    df_test = pd.read_csv(os.path.join(datapath, "test.csv"))
    train_test_cut = df_train.shape[0]
    print("Train shape : ", df_train.shape)
    print("Test shape : ", df_test.shape)
    # concat text data into single dataframe
    df_all = pd.concat(
        [df_train[['question_text']], df_test[['question_text']]],
        axis=0).reset_index(drop=True)
    # transform
    X_features = module.transform(df_all['question_text'])
    X_train = X_features[:train_test_cut]
    X_test = X_features[train_test_cut:]
    print('Preping took {:.2f}'.format(time.time() - t0))
    return df_train, df_test, X_train, X_test


def create_submission(X_train, y_train, X_test, df_test, thres,
                      module, filepath='../data/submission.csv'):
    """
    train model with entire training data, predict test data,
    and create submission file

    Parameters
    ----------
    X_train, y_train, X_test: features and targets

    df_test: dataframe, test data

    thres: float, a decision threshold for classification

    module: a python module

    filepath: tmp path to store score csv
    """
    # get model
    model = module.get_model()
    # train model
    t0 = time.time()
    print('Start to train model')
    model = model.fit(X_train, y_train)
    print('Training took {:.2f}'.format(time.time() - t0))
    # predict
    print('Start to predict')
    y_pred = (model.predict(X_test) > thres).astype('int')
    # create submission file
    print('Save submission file to {}'.format(filepath))
    pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred}).to_csv(filepath)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Evaluation and Create Submission")
    parser.add_argument('--datapath', nargs='?', default='../data/',
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='model_v0',
                        help='model version')
    parser.add_argument('--thres', type=int, default=0.5,
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
    df_train, df_test, X_train, X_test = load_and_preprocess(datapath, module)
    # 3. create submission file
    filepath = os.path.join(datapath, 'submit_' + model + '.csv')
    create_submission(X_train, df_train.target, X_test, df_test,
                      best_thres, module, filepath=filepath)
    print('Save submission file to {}.csv'.format(filepath))
    # record time spent
    print('All done and it took {:.2f}'.format(time.time() - t0))
