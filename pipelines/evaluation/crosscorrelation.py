import os
from typing import Dict

import numpy as np
import xarray as xr
from scipy import signal

from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.summary.summary_object import SummaryObjectTable


class CrossCorrelation(BaseSummary):
    """ Cross correlation summary module """

    def __init__(self, name: str = "crosscorrelation", resolution: float = 1.0):
        super().__init__(name)
        self.resolution = resolution

    def transform(self, file_manager: FileManager, y: xr.DataArray, y_hat: xr.DataArray, **kwargs) -> SummaryObjectTable:
        """ Perform cross correlation and return summary result """
        summary = SummaryObjectTable("crosscorrelation")
        result = crosscorr(y, y_hat, self.resolution)
        summary.set_kv('y_hat', result)
        return summary

    def get_params(self) -> Dict[str, object]:
        """ TODO """
        return {}

    def set_params(self):
        """ TODO """
        pass


def get_measures(x, lags):
    """
    Calculate mean and std for different lags
    """
    means = np.array([np.mean(x[-lag:x.size]) if lag <= 0 else np.mean(x[0:x.size - lag]) for lag in lags])
    stds = np.array([np.std(x[-lag:x.size]) if lag <= 0 else np.std(x[0:x.size - lag]) for lag in lags])
    stds = np.where(stds != 0, stds, np.nan)  # replace zeros by nan, we want to divide by stds
    return means, stds


def crosscorr(y, y_hat, resolution):
    """
    Calculate the cross-correlation of ground_truth and y for a given resolution
    """
    min_intersection = 200  # minimal number of data to calculate a coefficient from
    # calculate cross-correlation of ground_truth and y
    # note that signal.correlate returns correlation in convolution sense, i.e. if we want to get the pearson
    # correlation coefficient, we have to subtract the mean and divide by n and by the standard deviations
    # lag shifts the 2nd time series, so that only its tail hits the head of the 1st time series or the other way round

    # lag < 0 => analyze head of x, lag >= 0 => analyze tail of x
    def get_mean_std(x, my_lags, other_size):
        means = np.array([np.mean(x[0:np.minimum(x.size, other_size + lag)]) if lag < 0 else
                          np.mean(x[lag:(lag+min_length)]) for lag in my_lags])
        stds = np.array([np.std(x[0:np.minimum(x.size, other_size + lag)]) if lag < 0 else
                         np.std(x[lag:(lag+min_length)]) for lag in my_lags])
        stds = np.where(stds != 0, stds, np.nan)        # replace zeros by nan, because we want to divide by stds
        return means, stds

    lags = signal.correlation_lags(y.size, y_hat.size)
    min_length = np.minimum(y.size, y_hat.size)
    # lags are chosen maximal (from intersect 1 to maximal intersect to intersect 1 again)
    sizes = np.concatenate((np.arange(1, min_length), np.array([min_length]*(lags.size - 2 * (min_length - 1))),
                            np.flip(np.arange(1, min_length))))

    gt_means, gt_stds = get_mean_std(y, lags, y_hat.size)
    # use (-1) to essentially flip the order of the means and stds
    y_means, y_stds = get_mean_std(y_hat, (-1) * lags, y.size)
    # be careful: signal.correlate contains sums, whereby 2nd time series is shifted according to lags
    # => everything ok: first negative lags, heads of gt, tails of y; then positive lags, tails of gt, heads of y
    corr = (signal.correlate(y, y_hat) / sizes - y_means * gt_means) / (y_stds * gt_stds)
    # But do not return everything, trim output (at least min intersection, at most one year in both directions)
    max_lag_l = int(np.minimum(y_hat.size - min_intersection, 356 * 24 / resolution))
    max_lag_r = int(np.minimum(y.size - min_intersection, 356 * 24 / resolution))
    result = corr[(y_hat.size - 1 - max_lag_l):(y_hat.size - 1 + max_lag_r + 1)]
    return result


def eval(hparams):
    """
    Determine the cross-correlation for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'crosscorrelation'))

    #####
    # calculate cross-correlation between ground_truth and processed data (y, y_hat, y_hat_comp)
    ###
    CrossCorrelation(name='crosscorrelation')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
