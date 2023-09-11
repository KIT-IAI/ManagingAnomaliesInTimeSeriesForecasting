import os

import numpy as np

from pywatts_pipeline.core.pipeline import Pipeline

from pipelines.evaluation.autocorrelation import autocorr
from pywatts.summaries.metric_base import MetricBase


class AutoCorrelationMean(MetricBase):
    """
    Calculate the mean of the auto-correlations of ground_truth and y for a given resolution
    """

    def __init__(self, name: str = "autocorrelation_mean", resolution: float = 1.0):
        super().__init__(name)
        self.resolution = resolution

    def _apply_metric(self, p, t):
        result = autocorr_mean(t, p, self.resolution)
        return result


def autocorr_mean(ground_truth, y, resolution):
    """
    Calculate the mean of the auto-correlations of ground_truth and y for a given resolution
    """
    diffs = np.abs(autocorr(y, resolution) - autocorr(ground_truth, resolution))
    result = np.nanmean(diffs) if not np.all(np.isnan(diffs)) else np.nan
    return result


def eval(hparams):
    """
    Determine the mean of the auto-correlations of ground_truth and y for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'autocorrelation_mean'))

    #####
    # calculate the mean of the auto-correlations
    ###
    AutoCorrelationMean(name='autocorrelation_mean')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
