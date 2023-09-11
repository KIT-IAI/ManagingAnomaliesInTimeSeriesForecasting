import time

import xarray as xr

from pywatts_pipeline.core.summary.summary_formatter import SummaryJSON

import hooks.forecast_step_hook as hook
import pipelines.evaluation as evaluation_methods
import pipelines.forecasting.detected_anomalies_forecasting as detected_anomalies_forecasting_methods
import pipelines.forecasting.compensated_anomalies_forecasting as compensated_anomalies_forecasting_methods


def baseline_forecasting_step(hparams, train, test):
    """
    Experiment of baseline forecast using y for training and testing.
    """
    results = {}
    # obtain implemented forecast and evaluation methods
    forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
    eval_methods = [x for x in dir(evaluation_methods) if '__' not in x and x != 'pipelines']

    # apply each implemented forecast method
    for forecast_method in forecast_methods:
        print('Base ' + forecast_method + ' started')
        # load the forecasting pipeline
        forecast_pipeline = getattr(
            compensated_anomalies_forecasting_methods, forecast_method
        ).forecast(hparams)

        # train the forecasting pipeline
        forecast = forecast_pipeline.train({'y_target': train['y']})[0]['forecast']

        for forecast_horizon in range(24 * int(1 / hparams.resolution)):
            data = {
                'ground_truth': train['y'].loc[forecast.index][forecast_horizon:],
                'prediction': xr.DataArray(
                    forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                    dims=['index'], coords={'index': train['y'].loc[forecast.index][forecast_horizon:].index})
            }

            # evaluate forecast from training with each evaluation method
            for eval_method in eval_methods:
                evaluation_pipeline = getattr(
                    evaluation_methods, eval_method
                ).eval(hparams)
                _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                metric_name = list(eval_dict['Summary'].keys())[0]
                metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                results[f'train/{forecast_method}/{eval_method}/{forecast_horizon:03}/y'] = metric_value
                time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

        # forecast and evaluate test set
        if test is not None:
            forecast = forecast_pipeline.test({'y_target': test['y']})[0]['forecast']
            for forecast_horizon in range(24 * int(1 / hparams.resolution)):
                data = {
                    'ground_truth': test['y'].loc[forecast.index][forecast_horizon:],
                    'prediction': xr.DataArray(
                        forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                        dims=['index'], coords={'index': test['y'].loc[forecast.index][forecast_horizon:].index})
                }

                # evaluate forecast with each evaluation method
                for eval_method in eval_methods:
                    evaluation_pipeline = getattr(
                        evaluation_methods, eval_method
                    ).eval(hparams)
                    _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                    metric_name = list(eval_dict['Summary'].keys())[0]
                    metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                    metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                    results[f'{forecast_method}/{eval_method}/{forecast_horizon:03}/y'] = metric_value
                    time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

            if hparams.logging:
                forecast.to_dataframe('prediction').reset_index().to_csv(
                    f'{hparams.logging}_{forecast_method}_test.csv', index=False)

    return results


def raw_forecasting_step(hparams, train, test):
    """
    Experiment of raw forecast using y_hat for training and testing.
    """
    results = {}
    # obtain implemented forecast and evaluation methods
    forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
    eval_methods = [x for x in dir(evaluation_methods) if '__' not in x and x != 'pipelines']

    # apply each implemented forecast method
    for forecast_method in forecast_methods:
        print('Raw ' + forecast_method + ' started')
        # load the forecasting pipeline
        forecast_pipeline = getattr(
            compensated_anomalies_forecasting_methods, forecast_method
        ).forecast(hparams)

        # train the forecasting pipeline
        forecast = forecast_pipeline.train({'y_target': train['y_hat']})[0]['forecast']

        for forecast_horizon in range(24 * int(1 / hparams.resolution)):
            data = {
                'ground_truth': train['y'].loc[forecast.index][forecast_horizon:],
                'prediction': xr.DataArray(
                    forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                    dims=['index'], coords={'index': train['y'].loc[forecast.index][forecast_horizon:].index})
            }

            # evaluate forecast from training with each evaluation method
            for eval_method in eval_methods:
                evaluation_pipeline = getattr(
                    evaluation_methods, eval_method
                ).eval(hparams)
                _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                metric_name = list(eval_dict['Summary'].keys())[0]
                metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                results[f'train/{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat'] = metric_value
                time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

        # forecast and evaluate test set
        if test is not None:
            forecast = forecast_pipeline.test({'y_target': test['y_hat']})[0]['forecast']
            for forecast_horizon in range(24 * int(1 / hparams.resolution)):
                data = {
                    'ground_truth': test['y'].loc[forecast.index][forecast_horizon:],
                    'prediction': xr.DataArray(
                        forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                        dims=['index'], coords={'index': test['y'].loc[forecast.index][forecast_horizon:].index})
                }

                # evaluate forecast with each evaluation method
                for eval_method in eval_methods:
                    evaluation_pipeline = getattr(
                        evaluation_methods, eval_method
                    ).eval(hparams)
                    _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                    metric_name = list(eval_dict['Summary'].keys())[0]
                    metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                    metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                    results[f'{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat'] = metric_value
                    time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

            if hparams.logging:
                forecast.to_dataframe('prediction').reset_index().to_csv(
                    f'{hparams.logging}_{forecast_method}_test.csv', index=False)

    return results


def detected_anomalies_forecasting_step(hparams, train, test):
    """
    Experiment of forecast using y_hat and information on detected anomalies for training and y_hat for testing.
    """
    results = {}
    # obtain implemented forecast and evaluation methods
    forecast_methods = [x for x in dir(detected_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
    eval_methods = [x for x in dir(evaluation_methods) if '__' not in x and x != 'pipelines']

    # apply each implemented forecast method
    for forecast_method in forecast_methods:
        print('Detect ' + forecast_method + ' started')
        # load the forecasting pipeline
        forecast_pipeline = getattr(
            detected_anomalies_forecasting_methods, forecast_method
        ).forecast(hparams)
        # train the forecasting pipeline
        forecast = forecast_pipeline.train(train)[0]['forecast']

        for forecast_horizon in range(24 * int(1 / hparams.resolution)):
            data = {
                'ground_truth': train['y'].loc[forecast.index][forecast_horizon:],
                'prediction': xr.DataArray(
                    forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                    dims=['index'], coords={'index': train['y'].loc[forecast.index][forecast_horizon:].index})
            }

            # evaluate forecast from training with each evaluation method
            for eval_method in eval_methods:
                evaluation_pipeline = getattr(
                    evaluation_methods, eval_method
                ).eval(hparams)
                _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                metric_name = list(eval_dict['Summary'].keys())[0]
                metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                results[f'train/{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat'] = metric_value
                time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

        # forecast and evaluate test set
        if test is not None:
            forecast = forecast_pipeline.test(test)[0]['forecast']
            for forecast_horizon in range(24 * int(1 / hparams.resolution)):
                data = {
                    'ground_truth': test['y'].loc[forecast.index][forecast_horizon:],
                    'prediction': xr.DataArray(
                        forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                        dims=['index'], coords={'index': test['y'].loc[forecast.index][forecast_horizon:].index})
                }

                # evaluate forecast with each evaluation method
                for eval_method in eval_methods:
                    evaluation_pipeline = getattr(
                        evaluation_methods, eval_method
                    ).eval(hparams)
                    _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                    metric_name = list(eval_dict['Summary'].keys())[0]
                    metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                    metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                    results[f'{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat'] = metric_value
                    time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

            if hparams.logging:
                forecast.to_dataframe('prediction').reset_index().to_csv(
                    f'{hparams.logging}_{forecast_method}_test.csv', index=False)

    return results


def compensated_anomalies_forecasting_step(hparams, train, test):
    """
    Experiment of forecast using y_hat_comp for training and testing.
    """
    results = {}
    # obtain implemented forecast and evaluation methods
    forecast_methods = [x for x in dir(compensated_anomalies_forecasting_methods) if '__' not in x and x != 'pipelines']
    eval_methods = [x for x in dir(evaluation_methods) if '__' not in x and x != 'pipelines']

    # apply each implemented forecast method
    for forecast_method in forecast_methods:
        print('Compensate ' + forecast_method + ' started')
        # load the forecasting pipeline
        forecast_pipeline = getattr(
            compensated_anomalies_forecasting_methods, forecast_method
        ).forecast(hparams)

        # train the forecasting pipeline
        forecast = forecast_pipeline.train({'y_target': train['y_hat_comp']})[0]['forecast']

        for forecast_horizon in range(24 * int(1 / hparams.resolution)):
            data = {
                'ground_truth': train['y'].loc[forecast.index][forecast_horizon:],
                'prediction': xr.DataArray(
                    forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                    dims=['index'], coords={'index': train['y'].loc[forecast.index][forecast_horizon:].index})
            }

            # evaluate forecast from training with each evaluation method
            for eval_method in eval_methods:
                evaluation_pipeline = getattr(
                    evaluation_methods, eval_method
                ).eval(hparams)
                _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                metric_name = list(eval_dict['Summary'].keys())[0]
                metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                results[f'train/{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat_comp'] = metric_value
                time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

        # forecast and evaluate test set
        if test is not None:
            forecast = forecast_pipeline.test({'y_target': test['y_hat_comp']})[0]['forecast']
            for forecast_horizon in range(24 * int(1 / hparams.resolution)):
                data = {
                    'ground_truth': test['y'].loc[forecast.index][forecast_horizon:],
                    'prediction': xr.DataArray(
                        forecast[:forecast.shape[0] - forecast_horizon, forecast_horizon].values,
                        dims=['index'], coords={'index': test['y'].loc[forecast.index][forecast_horizon:].index})
                }

                # evaluate forecast with each evaluation method
                for eval_method in eval_methods:
                    evaluation_pipeline = getattr(
                        evaluation_methods, eval_method
                    ).eval(hparams)
                    _, eval_dict = evaluation_pipeline.train(data, summary=True, summary_formatter=SummaryJSON())
                    metric_name = list(eval_dict['Summary'].keys())[0]
                    metric_key = list(eval_dict['Summary'][metric_name]['results'].keys())[0]
                    metric_value = eval_dict['Summary'][metric_name]['results'][metric_key]
                    results[f'{forecast_method}/{eval_method}/{forecast_horizon:03}/y_hat_comp'] = metric_value
                    time.sleep(1)  # HOTFIX: Otherwise pyWATTS crashing because of "directory already exists"

            if hparams.logging:
                forecast.to_dataframe('prediction').reset_index().to_csv(
                    f'{hparams.logging}_{forecast_method}_test.csv', index=False)

    return results


def forecast_step(hparams, train, test):
    """
    Select forecast of respective experiment.
    """
    if hparams.experiment == 'forecast_with_compensated_anomalies':
        results = compensated_anomalies_forecasting_step(hparams, train, test)
    elif hparams.experiment == 'forecast_with_detected_anomalies':
        results = detected_anomalies_forecasting_step(hparams, train, test)
    elif hparams.experiment == 'raw_forecast':
        results = raw_forecasting_step(hparams, train, test)
    elif hparams.experiment == 'baseline_forecast':
        results = baseline_forecasting_step(hparams, train, test)

    # call hooks
    if hparams.hooks:
        hook.output(hparams, results)

    return results
