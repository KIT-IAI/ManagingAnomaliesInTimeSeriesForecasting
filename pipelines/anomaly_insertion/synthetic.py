import os

from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import FunctionModule

from pywatts.modules.generation import PowerAnomalyGeneration
from pywatts.modules.generation.unusual_behaviour_generation import UnusualBehaviour


def insert(hparams):
    """
    Insert anomalies of types 1 to 4 from the group of technical faults defined in
     M. Turowski, M. Weber, O. Neumann, B. Heidrich, K. Phipps, H. K. Çakmak, R. Mikut, and V. Hagenmeyer (2022).
     “Modeling and Generating Synthetic Anomalies for Energy and Power Time Series”. In: The Thirteenth ACM
     International Conference on Future Energy Systems (e-Energy ’22). ACM, pp. 471–484. doi: 10.1145/3538637.3539760
    or types 1 to 4 from the group of unusual consumption defined as types 5 to 8 in
     M. Turowski, B. Heidrich, K. Phipps, K. Schmieder, O. Neumann, R. Mikut, and V. Hagenmeyer (2022). “Enhancing
     Anomaly Detection Methods for Energy Time Series Using Latent Space Data Representations”. In: The Thirteenth ACM
     International Conference on Future Energy Systems (e-Energy ’22). ACM, pp. 208–227. doi: 10.1145/3538637.3538851
    in a given power time series
    """
    pipeline = Pipeline(path=os.path.join('run', hparams.name, 'insertion', 'synthetic'))
    seed = hparams.seed

    if hparams.anomaly_group == 'technical':
        anomaly_type1 = PowerAnomalyGeneration(
            'y_hat', anomaly='type1', count=hparams.type1, label=1, seed=seed + 1,
            length_params={
                'distribution': 'uniform',
                'min': 3,
                'max': 4465
            },
            anomaly_params={
                'k': hparams.k
            }
        )(x=pipeline['y'], labels=None)

        anomaly_type2 = PowerAnomalyGeneration(
            'y_hat', anomaly='type2', count=hparams.type2, label=2, seed=seed + 2,
            length_params={
                'distribution': 'uniform',
                'min': 1731,
                'max': 7434,
            },
            anomaly_params={
                'softstart': False
            }
        )(x=anomaly_type1['y_hat'], labels=anomaly_type1['labels'])

        anomaly_type3 = PowerAnomalyGeneration(
            'y_hat', anomaly='type3', count=hparams.type3, label=31, seed=seed + 3,
            anomaly_params={
                'is_extreme': False,
                'range_r': (0.61, 1.62),
            }
        )(x=anomaly_type2['y_hat'], labels=anomaly_type2['labels'])

        anomaly_type4 = PowerAnomalyGeneration(
            'y_hat', anomaly='type4', count=hparams.type4, label=4, seed=seed + 4,
            anomaly_params={
                'range_r': (1.15, 8.1)
            }
        )(x=anomaly_type3['y_hat'], labels=anomaly_type3['labels'])
    elif hparams.anomaly_group == 'unusual':
        anomaly_type1 = UnusualBehaviour(
            'y_hat', anomaly='type1', count=hparams.type1, label=1, seed=seed + 1,
            length_params={
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        )(x=pipeline['y'], labels=None)

        anomaly_type2 = UnusualBehaviour(
            'y_hat', anomaly='type2', count=hparams.type2, label=2, seed=seed + 2,
            length_params={
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        )(x=anomaly_type1['y_hat'], labels=anomaly_type1['labels'])

        anomaly_type3 = UnusualBehaviour(
            'y_hat', anomaly='type3', count=hparams.type3, label=31, seed=seed + 3,
            length_params={
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        )(x=anomaly_type2['y_hat'], labels=anomaly_type2['labels'])

        anomaly_type4 = UnusualBehaviour(
            'y_hat', anomaly='type4', count=hparams.type4, label=4, seed=seed + 4,
            length_params={
                'distribution': 'uniform',
                'min': 48,
                'max': 96 + 48
            }
        )(x=anomaly_type3['y_hat'], labels=anomaly_type3['labels'])

    FunctionModule(lambda x: x, name='y')(x=pipeline['y'])
    FunctionModule(lambda x: x, name='anomalies')(x=anomaly_type4['labels'])
    FunctionModule(lambda x: x, name='y_hat')(x=anomaly_type4['y_hat'])

    return pipeline
