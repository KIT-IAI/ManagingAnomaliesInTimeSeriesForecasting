import os

import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.metric_base import MetricBase

from pipelines.evaluation.crosscorrelation import crosscorr


class CrossCorrelationMean(MetricBase):
    """ Cross correlation mean summary module """

    def __init__(self, name: str = "crosscorrelation_mean", resolution: float = 1.0):
        super().__init__(name)
        self.resolution = resolution

    def _apply_metric(self, p, t):
        result = crosscorr(t,p,self.resolution)
        return 0


def crosscorr_mean(ground_truth, y, resolution):
    """
    Calculate the mean of the cross-correlations of ground_truth and y for a given resolution
    """
    vals = np.abs(crosscorr(ground_truth, y, resolution))
    result = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
    return xr.DataArray([result], dims=['index'], coords={
            'index': pd.date_range('2000', periods=1)
        })


def eval(hparams):
    """
    Determine the mean of the cross-correlations of ground_truth and y for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'crosscorrelation_mean'))

    #####
    # calculate cross-correlation between ground_truth and processed data (y, y_hat, y_hat_comp)
    ###
    CrossCorrelationMean(name='crosscorrelation_mean')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
