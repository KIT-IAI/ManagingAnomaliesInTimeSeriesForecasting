import os
import logging

import pandas as pd
import numpy as np

import hooks.loading_step_hook as hook


def loading_step(hparams):
    """
    Load data as panda dataframe and split into train and test set.
    """
    # load y data
    dataset = pd.read_csv(
        hparams.csv_path, index_col=hparams.time, parse_dates=True,
        delimiter=hparams.csv_separator, decimal=hparams.csv_decimal
    )

    # rename target index    
    rename_dict = {}
    rename_dict[hparams.target] = 'y'
    dataset.rename(columns=rename_dict, inplace=True)
    dataset.index.name = "index"

    # load y_hat and anomalies data
    dataset_y_hat = pd.read_csv(
        hparams.csv_path_y_hat, index_col=hparams.time, parse_dates=True,
        delimiter=hparams.csv_separator, decimal=hparams.csv_decimal
    )
    dataset['y_hat'] = dataset_y_hat[hparams.target]
    dataset['anomalies'] = dataset_y_hat['anomalies']
    dataset['anomalies'] = dataset['anomalies'].replace([2, 3, 4], 1)

    # load anomalies_hat data
    dataset_anomalies_hat = pd.read_csv(
        hparams.csv_path_anomalies_hat, index_col="time", parse_dates=True,
        delimiter=hparams.csv_separator, decimal=hparams.csv_decimal
    )
    dataset['anomalies_hat'] = dataset_anomalies_hat['0']

    # cut data to consider prior application of AnomalINN (supervised AnomalINN uses the first 15000 (UCI) or 8831 (CN)
    # data points for training and 4x24 = 96 samples; unsupervised only 96 samples)
    if hparams.detection_type == 'supervised':
        if '449' in hparams.csv_path:
            dataset.drop(dataset.index[:8831], inplace=True)
        else:
            dataset.drop(dataset.index[:15096], inplace=True)
    elif hparams.detection_type == 'unsupervised':
        dataset.drop(dataset.index[:96], inplace=True)

    dataset['anomalies_hat'] = dataset['anomalies_hat'].astype(np.int64)

    # load test csv file or split train
    if hparams.test != 'None' and float(hparams.test) != 0.0:
        try:
            split = 1-float(hparams.test)
            length = len(dataset)
            train = dataset.iloc[:int(split * length), :]
            test = dataset.iloc[int(split * length):, :]
            train = train[['y', 'y_hat', 'anomalies', 'anomalies_hat']]
            test = test[['y', 'y_hat', 'anomalies', 'anomalies_hat']]
        except:
            train = dataset[['y', 'y_hat', 'anomalies', 'anomalies_hat']]
            test = pd.read_csv(hparams.test, index_col=hparams.time, parse_dates=True)
            test.rename(columns=rename_dict, inplace=True)
    else:
        train = dataset[['y', 'y_hat', 'anomalies', 'anomalies_hat']]
        test = None

    # log descriptive statistics of loaded data
    logging.info('Descriptive statistics of loaded data:')
    logging.info(train.describe())
    if test is not None:
        logging.info(test.describe())
    logging.info('\n')

    # convert pandas dataframe to xarray
    train = {'y': train.to_xarray()['y'],
             'y_hat': train.to_xarray()['y_hat'],
             'anomalies': train.to_xarray()['anomalies'],
             'anomalies_hat': train.to_xarray()['anomalies_hat']}
    if test is not None:
        test = {'y': test.to_xarray()['y'],
                'y_hat': test.to_xarray()['y_hat'],
                'anomalies': test.to_xarray()['anomalies'],
                'anomalies_hat': test.to_xarray()['anomalies_hat']}

    # create hparams.name directory before calling hook
    # because of missing directory to save plots to
    os.makedirs(os.path.join('run', hparams.name), exist_ok=True)

    # call hooks
    if hparams.hooks:
        hook.output(hparams, train, test)

    return train, test
