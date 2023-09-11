import os

from utils.plots import data_plots, daily_load_plots


def output(hparams, train, test):
    """
    Print and plot information after the data loading
    """
    # Plot overall loaded train set of y
    data_plots(train['y'], name=os.path.join('run', hparams.name, f'load_train_y.png'))
    # Plot daily load curves of loaded train set for y
    daily_load_plots(train['y'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
        'run', hparams.name, 'load_train'))
    # Plot overall loaded test set of y and the corresponding daily load curves
    if test is not None:
        data_plots(test['y'], name=os.path.join('run', hparams.name, f'load_test_y.png'))
        daily_load_plots(test['y'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
            'run', hparams.name, 'load_test'))
