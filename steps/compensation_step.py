import logging

import numpy as np

import hooks.compensation_step_hook as hook
import pipelines.anomaly_compensation as methods


def compensation_step(hparams, train, test=None):
    """
    Compensate anomalies in the train and test set.
    """
    # if none compensation method is selected, log its skipping
    if hparams.compensation_method not in dir(methods):
        logging.info('Skipping anomaly compensation step.')
        return None, train, test

    # get compensation method
    compensation_method = getattr(methods, hparams.compensation_method).compensate(hparams)

    # apply compensation method to y_hat of train set
    train['y_hat_comp'] = train['y_hat'].copy().astype(np.float)
    train['y_hat_comp'][train['anomalies_hat'].values != 0] = np.nan
    train['y_hat_comp'] = compensation_method.train(train)[0]['y_hat_comp']

    # apply compensation method to y_hat of test set if existing
    if test is not None:
        test['y_hat_comp'] = test['y_hat'].copy().astype(np.float)
        test['y_hat_comp'][test['anomalies_hat'].values != 0] = np.nan
        test['y_hat_comp'] = compensation_method.test(test)[0]['y_hat_comp']

    # call hooks
    if hparams.hooks:
        hook.output(hparams, train, test)

    return None, train, test
