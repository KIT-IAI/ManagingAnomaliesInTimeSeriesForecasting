import os
from typing import Dict

import numpy as np
import xarray as xr

from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.core.summary_object import SummaryObjectList
from pywatts.summaries.metric_base import MetricBase


class TemplateNumpy2Value(MetricBase):
    """
    TODO: describe metric
    """

    def _apply_metric(self, p, t):
        # TODO: Replace return value by metric result (for single value metrics)
        return 0


class TemplateDataArray2List(BaseSummary):
    """
    TODO: describe metric
    """

    def __init__(self, name: str = "template"):
        super().__init__(name)

    def transform(self, file_manager: FileManager, y: xr.DataArray, y_hat: xr.DataArray, **kwargs) -> SummaryObjectList:
        summary = SummaryObjectList("template")
        result = [1,2,3]
        summary.set_kv('y_hat', result)
        return summary

    def get_params(self) -> Dict[str, object]:
        return {}

    def set_params(self):
        pass


def eval(hparams):
    """
    TODO: describe pipeline
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'evaluation', 'template'))

    #####
    # TODO: define pipeline to calculate metric summaries
    ###
    TemplateDataArray2List(name='template')(
        y=pipeline['ground_truth'], y_hat=pipeline['prediction']
    )

    return pipeline
