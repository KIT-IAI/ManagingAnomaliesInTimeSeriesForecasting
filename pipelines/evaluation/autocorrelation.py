import os
from typing import Dict

import numpy as np
import xarray as xr
from scipy import signal

from pywatts_pipeline.core.pipeline import Pipeline

from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList

from pywatts_pipeline.core.util.filemanager import FileManager


class AutoCorrelation(BaseSummary):
    """
    Calculate the auto-correlation of y for a given resolution
    """

    def __init__(self, name: str = "template", resolution: float = 1.0):
        super().__init__(name)
        self.resolution = resolution

    def transform(self, file_manager: FileManager, y: xr.DataArray, y_hat: xr.DataArray, **kwargs) -> SummaryObjectList:
        summary = SummaryObjectList("autocorrelation")
        result = autocorr(y_hat, self.resolution)
        summary.set_kv('y_hat', result.tolist())
        return summary

    def get_params(self) -> Dict[str, object]:
        return {}

    def set_params(self):
        pass

def autocorr(x, resolution):
    """
    Calculate the auto-correlation of y for a given resolution
    """
    min_intersection = 100  # minimal number of data to calculate a coefficient from
    # note that signal.correlate returns correlation in convolution sense, i.e. if we want to get the pearson
    # correlation coefficient, we have to subtract the mean and divide by n and by the standard deviations
    lags = signal.correlation_lags(x.size, x.size)
    sizes = x.size - np.abs(lags)
    means = np.array([np.mean(x[-lag:x.size]) if lag <= 0 else np.mean(x[0:x.size - lag]) for lag in lags])
    stds = np.array([np.std(x[-lag:x.size]) if lag <= 0 else np.std(x[0:x.size - lag]) for lag in lags])
    stds = np.where(stds != 0, stds, np.nan)  # replace zeros by nan, we want to divide by stds
    # auto correlation is symmetric in lags, instead of lag in [-(n-1),..., 0,...,(n-1)]
    # only return lag in [-(n-1),...,0] = [n-1,...,0], hence flip array

    # But we do not want to return everything, we want lags not to exceed 1 year and the distance from the maximal lag
    # to length of the time series should be at least min_intersection (max_lag is last lag that is considered)
    max_lag = int(np.minimum(x.size - min_intersection, 356 * 24 / resolution))
    result = np.flip(((signal.correlate(x, x) / sizes - means * np.flip(means)) /
                    (stds * np.flip(stds)))[(x.size - 1 - max_lag):x.size])
    return result


def eval(hparams):
    """
    Determine the auto-correlation for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'autocorrelation'))

    #####
    # calculate auto-correlation
    ###
    AutoCorrelation(name='autocorrelation')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
