import logging

import hooks.detection_step_hook as hook
import pipelines.anomaly_detection as methods


def detection_step(hparams, train, test=None):
    """
    Detect anomalies in the train and test set.
    """
    # if none detection method is selected, log its skipping
    if hparams.detection_method not in dir(methods):
        logging.info('Skipping anomaly detection step.')
        return None, train, test

    # get detection method
    detection_method = getattr(methods, hparams.detection_method).detect(hparams)

    # apply detection method to train set
    train['anomalies_hat'] = detection_method.train(train)[0]['anomalies_hat']

    # apply detection method to detect to test set if existing
    if test is not None:
        test['anomalies_hat'] = detection_method.test(test)[0]['anomalies_hat']

    # call hooks
    if hparams.hooks:
        hook.output(hparams, train, test)

    return None, train, test
