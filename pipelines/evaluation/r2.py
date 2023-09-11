import os

from sklearn.metrics import r2_score

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.summaries.metric_base import MetricBase


class R2(MetricBase):
    """
    R2 summary module
    """

    def _apply_metric(self, p, t):
        result = r2_score(y_true=t, y_pred=p)
        return result


def eval(hparams):
    """
    Determine the R2 score for a given dataset.
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'R2'))

    #####
    # calculate R2
    ###
    R2(name='r2')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
