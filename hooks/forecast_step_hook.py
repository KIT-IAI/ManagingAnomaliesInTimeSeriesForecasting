import numpy as np
import os

import pipelines.evaluation as evaluation_methods
import pipelines.forecasting.detected_anomalies_forecasting as detected_anomalies_forecasting_methods
import pipelines.forecasting.compensated_anomalies_forecasting as compensated_anomalies_forecasting_methods
from utils.plots import result_plot, correlation_histograms, correlation_plot


def output(hparams, results):
    """
    Print and plot information after the forecast
    """
    # Print forecast results
    print('Forecast step:')
    print(f'results - {results.keys()}')
    print()
    for key in results:
        if np.isscalar(results[key]):
            print(f'{key} - {results[key]}')
        else:
            print(f'{key} - ARRAY')

    print()

    # if results is not empty, create plots
    if results:
        if hparams.experiment.lower() == 'forecast_with_detected_anomalies':
            forecast_methods = [x for x in dir(detected_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
        elif hparams.experiment.lower() == 'forecast_with_compensated_anomalies':
            forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
        elif hparams.experiment.lower() == 'raw_forecast':
            forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
        elif hparams.experiment.lower() == 'baseline_forecast':
            forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
