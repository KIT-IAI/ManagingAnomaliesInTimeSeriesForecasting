import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts.modules import FunctionModule

import pipelines.forecasting.compensated_anomalies_forecasting as forecasting_methods


def compensate(hparams):
    """ Pipeline to compensate anomalies by forecasting """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'compensation', 'forecasting'))

    # Define forecast
    forecast_pipeline = getattr(
        forecasting_methods, hparams.detect_forecasting_method
    ).forecast(hparams)

    forecast_pipeline(y_target=pipeline['y'], computation_mode=ComputationMode.Train)
    forecast = forecast_pipeline(y_target=pipeline['y_hat'], computation_mode=ComputationMode.Transform)

    # Determine anomalies given a forecast
    FunctionModule(lambda y: y, name='y_hat_comp')(y=forecast['forecast'])
    return pipeline
