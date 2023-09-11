import os

import numpy as np

from utils.plots import daily_load_plots, data_plots, plot_overview, correlation_histograms
# from pipelines.evaluation.crosscorrelation import crosscorr


def output(hparams, train, test):
    """
    Print and plot information after the anomaly compensation
    """
    # Print information on compensation step
    print('Compensation step:')
    print(f'train - {train.keys()}')
    if test is not None:
        print(f'test - {test.keys()}')
    print()

    # Plot comparison of y and y_hat_comp as well as y_hat and y_hat_comp
    # TODO: adapt plot
    if 'y' in train and 'y_hat' in train and 'y_hat_comp' in train:
        data_plots(np.abs(train['y'] - train['y_hat_comp']), np.abs(train['y_hat'] - train['y_hat_comp']),
                   name=os.path.join('run', hparams.name, 'compareTrainingData.png'), loc='upper right',
                   labels=['abs diff of y and y_hat_comp', 'abs diff of y_hat and y_hat_comp'])

    # Plot daily load curves of y_hat_comp for the train set
    daily_load_plots(train['y_hat_comp'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
        'run', hparams.name, 'compensate_train'))
    if test is not None and 'y_hat_comp' in test:
        daily_load_plots(test['y_hat_comp'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
            'run', hparams.name, 'compensate_test'))

    if 'y_hat' in train and 'y_hat' in test:
        plot_overview(train, test, os.path.join('run', hparams.name, 'data_overview.png'))
