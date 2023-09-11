import os

import numpy as np

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.metric_base import MetricBase


class SNR(MetricBase):
    """
    snr summary module
    """

    def _apply_metric(self, p, t):
        result = snr(p)
        return result


def snr(data):
    """
    Calculate SNR (signal-to-noise ratio) as ratio of mean and standard deviation of the given time series
    """
    std = np.std(data)
    if std == 0.0:
        result = -1
    else:
        result = (np.mean(data) ** 2) / (np.std(data) ** 2)

    return result


def eval(hparams):
    """
    Determine the SNR (signal-to-noise ratio) for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'SNR'))

    #####
    # calculate SNR
    ###
    SNR(name='snr')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
