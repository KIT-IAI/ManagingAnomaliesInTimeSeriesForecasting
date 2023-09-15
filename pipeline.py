import os

import argparse
import logging
import pprint

import numpy as np
import pandas as pd

from steps.compensation_step import compensation_step
from steps.forecast_step import forecast_step
from steps.loading_step import loading_step


def str2intorfloat(v):
    try:
        return int(v)
    except:
        return float(v)


def parse_hparams(args=None):
    """
    Parses command line statement

    :return: the parsed arguments
    """
    # prepare argument parser
    parser = argparse.ArgumentParser(
        description='Anomaly detection and compensation pipeline for time series forecasting.'
    )
    # run_name
    parser.add_argument('name', type=str, help='Name of the run.')
    # csv path file
    parser.add_argument('csv_path', type=str, help='Path to the data CSV file.')
    # time_index
    parser.add_argument('time', type=str, help='Name of the time index.')
    # data_index
    parser.add_argument('target', type=str, help='Name of the target index.')

    # csv path file y_hat and anomalies
    parser.add_argument('csv_path_y_hat', type=str, help='Path to the y_hat data CSV file.')
    # csv path file anomalies_hat
    parser.add_argument('csv_path_anomalies_hat', type=str, help='Path to the anomalies_hat data CSV file.')

    # experiment
    parser.add_argument('--experiment', choices=['forecast_with_compensated_anomalies',
                                                 'forecast_with_detected_anomalies', 'raw_forecast',
                                                 'baseline_forecast'],
                        default='forecast_with_compensated_anomalies',
                        help='Experiment to be run ("forecast_with_compensated_anomalies", '
                             '"forecast_with_detected_anomalies", "raw_forecast", "baseline_forecast", or '
                             '"robust_forecast").')
    # data cut for supervised or unsupervised anomaly detection
    parser.add_argument('--detection_type', choices=['supervised', 'unsupervised'], default='unsupervised',
                        help='Type of applied detection to determine right removal of cINN or cVAE training data')
    # test split or file
    parser.add_argument('--test', type=str, default='0.2',
                        help='Path to the test CSV file or float test split value.')
    # seed for pipeline run
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed to be used in the pipeline run.')
    # time resolution of input data
    parser.add_argument('--resolution', type=float, default=0.25,
                        help='Time resolution of the input data in hours (e.g. 0.25 for 15 minutes).')
    # hooks
    parser.add_argument('--hooks', action='store_true',
                        help='Turn on pipeline hooks for each step.')
    # debugging
    parser.add_argument('--debug', action='store_true',
                        help='Turn on pipeline debugging output.')

    # compensation_step
    parser.add_argument('--compensation_method', type=str, default='my_prophet',
                        help='Compensation method to be used (anomaly detections needed by \'anomalies\' index).')

    # forecast_step
    # parser.add_argument('--forecast_sample_size_NONE', type=int, default=24 * 4,
    #                     help='Sample size used in the training of certain forecast methods (default 96)')
    parser.add_argument('--forecast_arima_p', type=int, default=1,
                        help='Used p value in applied ARIMA forecast')
    parser.add_argument('--forecast_arima_d', type=int, default=1,
                        help='Used d value in applied ARIMA forecast')
    parser.add_argument('--forecast_arima_q', type=int, default=1,
                        help='Used q value in applied ARIMA forecast')

    # logging
    parser.add_argument('--logging', type=str, default=None,
                        help='Turn on CSV logging to the given path (e.g. Logs.csv).')

    # delimiter
    parser.add_argument('--csv_separator', type=str, default=',',
                        help='CSV file column separator (default ;).')
    # decimal
    parser.add_argument('--csv_decimal', type=str, default='.',
                        help='CSV file decimal delimiter (default ,).')

    # convert argument strings
    parsed_hparams = parser.parse_args(args=args)

    return parsed_hparams


def log_results(hparams, results_dict, filepath):
    if results_dict is not None:
        results = {}
        results.update(vars(hparams))
        results.update(results_dict)
        # don't log arrays
        for key in results.keys():
            if np.array(results[key]).size > 1:
                results[key] = 'ARRAY'
        pprint.pprint(results)
        results_frame = pd.DataFrame(results, index=[0])

        if os.path.exists(filepath):
            old_frame = pd.read_csv(filepath, sep=hparams.csv_separator, index_col=0)
            results_frame = pd.concat([old_frame, results_frame], ignore_index=True)
            print(results_frame)
        results_frame.to_csv(filepath, sep=hparams.csv_separator)


def run_pipeline(hparams):
    import time
    import random
    time.sleep(random.random() * 20)
    # load data
    train, test = loading_step(hparams)

    # compensate detected anomalies
    compensate_results, train, test = compensation_step(hparams, train, test)

    # forecast time series
    forecast_results = forecast_step(hparams, train, test)

    # log results into csv regarding hparams
    if hparams.logging is not None:
        log_results(hparams, compensate_results, f'{hparams.logging}_compensate.csv')
        log_results(hparams, forecast_results, f'{hparams.logging}_forecast.csv')

    print('Finished run of ' + hparams.name)


if __name__ == '__main__':
    # parse command line statement
    hparams = parse_hparams()

    # set logging level
    if hparams.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)
    logging.info(hparams)

    run_pipeline(hparams)
