import os

import numpy as np
import pyinform as d

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.metric_base import MetricBase


class Entropy(MetricBase):
    """ Entropy summary module """

    def _apply_metric(self, p, t):
        result = entropy(p)
        return result


def entropy(data):
    """
    Calculate the Shannon entropy
    """
    # constant series have entropy 0 (has to be treated, since bin_series cannot handle that)
    if np.max(data) - np.min(data) == 0:
        return 0
    # standardize data
    x = (data - np.mean(data)) / np.std(data)
    # calculate Shannon entropy of the empirical distribution
    # since load data is continuous, we need to make the time series discrete before we can apply block_entropy
    # this is achieved by binning
    result = d.block_entropy(d.utils.bin_series(x, step=0.1 * np.std(x))[0], k=1)
    return result


def eval(hparams):
    #####
    # calculate entropy
    ###
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'entropy'))
    Entropy(name='entropy')(
        y_hat=pipeline['prediction'], y=pipeline['ground_truth'])

    return pipeline
