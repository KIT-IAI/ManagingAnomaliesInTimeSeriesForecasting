import logging

import hooks.insertion_step_hook as hook
import pipelines.anomaly_insertion as methods


def insertion_step(hparams, train, test=None):
    """
    Insert anomalies to train and test set.
    """
    # if none insertion method is selected, log its skipping
    if hparams.insertion_method not in dir(methods):
        logging.info('Skipping anomaly insertion step.')
        return None, train, test

    # get insertion method
    insertion_method = getattr(methods, hparams.insertion_method).insert(hparams)

    # apply insertion method to train set
    train, _ = insertion_method.train(train)

    # apply insertion method to test set if existing
    if test is not None:
        test, _ = insertion_method.test(test)

    # call hooks
    if hparams.hooks:
        hook.output(hparams, train, test)

    return None, train, test
