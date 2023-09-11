import os

from pywatts_pipeline.core.pipeline import Pipeline

from pipelines.evaluation.crosscorrelation import crosscorr
from pywatts.summaries.metric_base import MetricBase


class Correlation(MetricBase):
    """
    correlation summary module
    """
    def __init__(self, name: str = "crosscorrelation", resolution: float = 1.0):
        super().__init__(name)
        self.resolution = resolution

    def _apply_metric(self, p, t):
        result = corr(t, p, self.resolution)
        return result


def corr(ground_truth, y, resolution):
    cross_c = crosscorr(ground_truth, y, resolution)
    # cross_c contains cross-correlation coefficients
    # for symmetric lags, i.e. for lags in [-n,-(n-1),...,0,...,n-1,n]
    # whereby cross_c.size = 2n+1 => cross_c[lag=0]
    # = cross_c[n+1] = cross_c[(cross_c.size - 1)/2]
    return cross_c[int((cross_c.size - 1) / 2)]


def eval(hparams):
    """ Calculation of the correlation on a given dataset. """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'correlation'))

    #####
    # correlation calculation of ground_truth and processed data (y, y_hat, y_hat_comp)
    ###
    Correlation(name='correlation')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
