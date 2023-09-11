import os

from utils.plots import daily_load_plots


def output(hparams, train, test):
    """
    Print and plot information after the anomaly insertion
    """
    # Print information of insertion
    print('Insert step:')
    print(f'train - {train.keys()}')
    if test is not None:
        print(f'test - {test.keys()}')
    print()

    # Plot daily load curves of y_hat for train set
    if 'y_hat' in train:
        daily_load_plots(train['y_hat'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
            'run', hparams.name, f'insert_train'))
    # Export train set as csv file
    for key in train:
        train[key].to_pandas().to_csv(os.path.join('run', hparams.name, f'insert_train_{key}.csv'))

    # Plot daily load curves of y_hat for test set and export test set as csv file
    if test is not None:
        if 'y_hat' in test:
            daily_load_plots(test['y'], width=int(24 * 1 / hparams.resolution), prefix=os.path.join(
                'run', hparams.name, f'insert_test_{key}'))
        for key in test:
            test[key].to_pandas().to_csv(os.path.join('run', hparams.name, f'insert_test_{key}.csv'))
