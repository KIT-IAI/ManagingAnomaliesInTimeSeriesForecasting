import logging
from typing import Dict, Tuple

import tensorflow
import numpy as np
import xarray as xr
from pywatts_pipeline.core.transformer.base import BaseEstimator
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers
from tensorflow.keras import activations, optimizers, initializers
from tensorflow import keras


class ProfileNeuralNetwork(BaseEstimator):
    """
    This module implements the profile neural network. It is a model for forecasting short-term electrical load.
    Therefore, it takes into account, trend information, calendar_information, historical input but also the profile
    of the load.
    Note the horizon is extracted from the data
    If you use it please cite:
        Benedikt Heidrich, Marian Turowski, Nicole Ludwig, Ralf Mikut, and Veit Hagenmeyer. 2020.
        Forecasting energy time series with profile neural networks. In Proceedings of the Eleventh ACM International
        Conference on Future Energy Systems (e-Energy ’20). Association for Computing Machinery, New York, NY, USA,
        220–230. DOI:https://doi.org/10.1145/3396851.3397683

    :param name: The name of the module
    :type name: str
    :param epochs: The number of epochs the model should be trained.
    :type epochs: int
    :param offset: The number of samples at the beginning of the dataset that should be **not** considered for training.
    :type offset: int
    :param batch_size: The batch size which should be used for training
    :type batch_size: int
    :param validation_split: The share of data which should be used for validation
    :type validation_split: float
    """

    def __init__(self, name: str = "PNN", epochs=50, offset=0, batch_size=512, validation_split=0.3,
                 early_stopping=None, refit_last_layer=False, return_all=False):
        super().__init__(name)
        self.epochs = epochs
        self.offset = offset
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.refit_last_layer = refit_last_layer
        self.return_all = return_all
        self.targets = {
            "target": self.name,
            "target2": "colorful_noise"
        }

    def refit(self, target, **kwargs):
        fc_weights = self.fc.get_weights()
        trend_weights = self.trend.get_weights()
        agg_weights_before = self.pnn.layers[-2].get_weights()
        historical_input, trend, profile, dummy_input = self._get_inputs(kwargs, 0)

        input, t, _ = self._clean_dataset({
            "hist_input": historical_input,
            "full_trend": trend,
            "profile": profile,
            "dummy_input": dummy_input
        }, target.values)

        if self.refit_last_layer:
            self.fc.trainable = False
            self.trend.trainable = False
            self.train_pnn.layers[-2].trainable = True
            self.train_pnn.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                                   metrics=[_root_mean_squared_error])
        self.train_pnn.fit(input, t, epochs=5, batch_size=self.batch_size)

        # Structure of weights: List of layers and its weights
        #   * its entry is a list of two elements, The first are the weights, the second the bias.

        agg_weights_after = self.pnn.layers[-2].get_weights()
        return f"## Weight updates:\n" \
               f" * Difference Colorful noise: {self.calculate_weight_difference(fc_weights, self.fc.get_weights())} by {self.fc.count_params()} parameters\n" \
               f" * Difference Trend: {self.calculate_weight_difference(trend_weights, self.trend.get_weights())} by {self.trend.count_params()} parameters\n" \
               f" * Aggregation Layer Before: {agg_weights_before} \n" \
               f" * Aggregation Layer After: {agg_weights_after} \n"

    def calculate_weight_difference(self, weights_before, weights_after):
        _weights = 0
        for w1, w2 in zip(weights_before, weights_after):
            _weights += np.sum(np.abs(w1 - w2))
        return _weights

    def get_params(self) -> Dict[str, object]:
        """ Get parameter for this object as dict.

        :return: Object parameters as json dict
        """
        # TODO early stopping
        return {
            "epochs": self.epochs,
            "offset": self.offset,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split
        }

    def set_params(self, epochs=None, offset=None, batch_size=None, validation_split=None):
        """
        :param epochs: The number of epochs the model should be trained.
        :type epochs: int
        :param offset: The number of samples at the beginning of the dataset that should be **not** considered for training.
        :type offset: int
        :param batch_size: The batch size which should be used for training
        :type batch_size: int
        :param validation_split: The share of data which should be used for validation
        :type validation_split: float
        """
        # TODO early stopping
        if batch_size:
            self.batch_size = batch_size
        if epochs:
            self.epochs = epochs
        if offset:
            self.offset = offset
        if validation_split:
            self.validation_split = validation_split

    def transform(self, **kwargs) -> xr.DataArray:
        """
        Forecast the electrical load for the given input.

        :param historical_input: The historical input
        :type historical_input: xr.DataArray
        :param calendar: The calendar information of the dates that should be predicted.
        :type calendar: xr.DataArray
        :param temperature: The temperature of the dates that should be predicted
        :type temperature: xr.DataArray
        :param humidity: The humidity of the dates that should be predicted
        :type humidity: xr.DataArray
        :param profile: The profile of the dates that should be predicted
        :type profile: xr.DataArray
        :param trend: The trend information of the dates that should be predicted
        :type trend: xr.DataArray
        :return: The prediction
        :rtype: xr.DataArray
        """
        reference = kwargs["historical_input"]
        historical_input, trend, profile, dummy_input = self._get_inputs(kwargs, 0)

        result = self.pnn.predict({
            "hist_input": historical_input,
            "full_trend": trend,
            "profile": profile,
            "dummy_input": dummy_input
        })
        if self.return_all:
            return {
                self.name: numpy_to_xarray(result[0], reference),
                "colorful_noise": numpy_to_xarray(result[1].reshape((-1, self.horizon)), reference),
                "profile": numpy_to_xarray(result[2].reshape((-1, self.horizon)), reference),
                "trend": numpy_to_xarray(result[3].reshape((-1, self.horizon)), reference),

            }
        return {self.name: numpy_to_xarray(result[0], reference),
                "colorful_noise": numpy_to_xarray(result[1].reshape((-1, self.horizon)), reference),
                }

    def fit(self, target, **kwargs):
        """
        Fit the Profile Neural Network.

        :param historical_input: The historical input
        :type historical_input: xr.DataArray
        :param calendar: The calendar information of the dates that should be predicted.
        :type calendar: xr.DataArray
        :param temperature: The temperature of the dates that should be predicted
        :type temperature: xr.DataArray
        :param humidity: The humidity of the dates that should be predicted
        :type humidity: xr.DataArray
        :param profile: The profile of the dates that should be predicted
        :type profile: xr.DataArray
        :param trend: The trend information of the dates that should be predicted
        :type trend: xr.DataArray
        :param target: The ground truth of the desired prediction
        :type target: xr.DataArray
        """
        self.horizon = target.shape[-1]
        historical_input, trend, profile, dummy_input = self._get_inputs(kwargs, self.offset)
        input_length = historical_input.shape[-1]
        trend_length = trend.shape[-1]
        self.train_pnn, self.fc, self.trend, self.profile, self.pnn = _PNN(self.horizon, n_steps_in=input_length,
                                                                           trend_length=trend_length,
                                                                           ext_shape=dummy_input.shape[1:])

        input_val, t_val, _ = self._clean_dataset({
                "hist_input": historical_input[int(len(historical_input) * (1 - self.validation_split)):],
                "full_trend": trend[int(len(historical_input) * (1 - self.validation_split)):],
                "profile": profile[int(len(historical_input) * (1 - self.validation_split)):],
                "dummy_input": dummy_input[int(len(historical_input) * (1 - self.validation_split)):]
            }, 
            target.values[self.offset:][int(len(historical_input) * (1 - self.validation_split)):]
        )

        input, t, _ = self._clean_dataset({
                "hist_input": historical_input[:int(len(historical_input) * (1 - self.validation_split))],
                "full_trend": trend[:int(len(historical_input) * (1 - self.validation_split))],
                "profile": profile[:int(len(historical_input) * (1 - self.validation_split))],
                "dummy_input": dummy_input[:int(len(historical_input) * (1 - self.validation_split))]
            },
            target.values[self.offset:][:int(len(historical_input) * (1 - self.validation_split))]
        )

        if self.early_stopping:
            self.train_pnn.fit(input, t, epochs=self.epochs, batch_size=self.batch_size,
                               validation_data=(input_val, [t_val]), callbacks=[self.early_stopping])

        else:
            self.train_pnn.fit(input, t, epochs=self.epochs, batch_size=self.batch_size,
                               validation_split=self.validation_split)
        self.is_fitted = True

    def clone_module(self, refit_last_layer=False, name="cloned"):
        pnn_module = ProfileNeuralNetwork(self.name + f"_{name}{'_last_layer' if refit_last_layer == True else ''}",
                                          refit_last_layer=refit_last_layer)

        pnn_module.return_all = self.return_all
        pnn_module.epochs = self.epochs
        pnn_module.offset = self.offset
        pnn_module.batch_size = self.batch_size
        pnn_module.validation_split = self.validation_split
        pnn_module.early_stopping = self.early_stopping
        pnn_module.horizon = self.horizon
        pnn_module.is_fitted = self.is_fitted

        pnn_module.fc = keras.models.clone_model(self.fc) if self.fc is not None else None
        pnn_module.fc.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                              metrics=[_root_mean_squared_error])
        pnn_module.fc.set_weights(self.fc.get_weights())

        pnn_module.trend = keras.models.clone_model(self.trend) if self.trend is not None else None
        pnn_module.trend.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                                 metrics=[_root_mean_squared_error])
        pnn_module.trend.set_weights(self.trend.get_weights())

        pnn_module.profile = keras.models.clone_model(self.profile) if self.profile is not None else None
        pnn_module.profile.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                                   metrics=[_root_mean_squared_error])
        pnn_module.profile.set_weights(self.profile.get_weights())

        pnn_module.train_pnn, pnn_module.pnn = get_agg_layer(*pnn_module.fc.inputs,
                                                             pnn_module.fc,
                                                             pnn_module.profile.input,
                                                             pnn_module.profile,
                                                             pnn_module.trend,
                                                             pnn_module.trend.input,
                                                             )

        pnn_module.pnn.layers[-2].set_weights(self.pnn.layers[-2].get_weights())
        return pnn_module

    def save(self, fm: FileManager) -> Dict:
        """
        Stores the PNN at the given path

        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        json = super().save(fm)
        if self.is_fitted:
            filepath = fm.get_path(f"{self.name}.h5")
            self.pnn.save(filepath=filepath)
            json.update({
                "pnn": filepath
            })
        return json

    @classmethod
    def load(cls, load_information) -> BaseEstimator:
        """
        Load the PNN model.

        :param params:  The paramters which should be used for restoring the PNN.
        :return: A wrapped keras model.
        """
        pnn_module = ProfileNeuralNetwork(name=load_information["name"], **load_information["params"])
        if load_information["is_fitted"]:
            try:
                pnn = keras.models.load_model(filepath=load_information["pnn"],
                                              custom_objects={"_sum_squared_error": _sum_squared_error,
                                                              "_root_mean_squared_error": _root_mean_squared_error})
            except Exception as exception:
                logging.error("No model found in %s.", load_information['pnn'])
                raise exception
            pnn_module.pnn = pnn
            pnn_module.is_fitted = True
        return pnn_module

    @staticmethod
    def _clean_dataset(X, y, y2=None, same_values_in_a_row=2):
        """
        Cleans the dataset. The following three rules are applied:
        Arguments:
            X: Input data
            y: Target data
            same_values_in_a_row: parameter which indicates how often the same value in a row is accetable
        """

        def _check_instance(data):
            """
                Checks if the data is nan, contains more than same_values_in_a_row values which has the same value and
                if the value is zero
                Returns: Bool: False if one of the conditions applies
            """
            counter = 0
            d_last = -1
            for d in data:
                if d_last == d:
                    if counter > same_values_in_a_row:
                        return False
                    counter += 1
                else:
                    counter = 0
                d_last = d
            return True

        x_cleaned = {}
        for x in X:
            x_cleaned[x] = []
        y_cleaned = []
        y2_cleaned = []
        for i in range(len(y)):
            if not np.any(np.isnan(X["hist_input"][i])) and not np.any(np.isnan(y[i])) and not np.any(
                    y[i] == 0) and _check_instance(X["hist_input"][i]) and _check_instance(y[i]) and not np.any(
                    np.isnan(X["full_trend"][i])) and not np.any(np.isnan(X["dummy_input"][i])):
                # add to cleaned dataset
                for key, x in X.items():
                    x_cleaned[key].append(x[i])
                y_cleaned.append(y[i, :])
                if y2 is not None:
                    y2_cleaned.append(y2[i, :])
        for key, x in x_cleaned.items():
            x_cleaned[key] = np.array(x)
        return x_cleaned, np.array(y_cleaned), np.array(y2_cleaned)

    def _get_inputs(self, kwargs, offset):
        historical_input = kwargs["historical_input"].values[offset:]
        trend = kwargs["trend"].values[offset:]
        profile = kwargs["profile"].values[offset:]
        del kwargs["historical_input"]
        del kwargs["trend"]
        del kwargs["profile"]
        dummy_input = np.concatenate(
            [data.values[offset:].reshape((len(historical_input), -1)) for data in kwargs.values()],
            axis=-1)
        return historical_input, trend, profile, dummy_input


def _sum_squared_error(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sum(K.square(y_true - y_pred), axis=-1)


def _PNN(n_steps_out, ext_shape, n_steps_in=36, trend_length=5) -> Tuple[
    keras.Model, keras.Model, keras.Model, keras.Model, keras.Model]:

    activation = activations.elu

    def hist_encoder():
        conv_input = keras.Input(shape=(n_steps_in,), name="hist_input")
        conv = layers.Reshape((n_steps_in, 1))(conv_input)
        conv = layers.Conv1D(4, [3], activation=activation, padding='same')(conv)
        conv = layers.MaxPool1D(pool_size=2)(conv)

        conv = layers.Conv1D(1, [7], activation=activation, padding='same')(conv)
        conv = layers.MaxPool1D(pool_size=2)(conv)
        conv = layers.Flatten()(conv)
        conv = layers.Dense(n_steps_out)(conv)
        conv = layers.Reshape((n_steps_out, 1))(conv)
        return conv, conv_input

    def external_encoder():
        dummy_input = keras.Input(shape=ext_shape, name="dummy_input")
        dummy = layers.Flatten()(dummy_input)
        dummy = layers.Dense(n_steps_out, activation=activation)(dummy)
        dummy = layers.Reshape((n_steps_out, 1))(dummy)
        return dummy, dummy_input

    def prediction_network():
        conv, conv_input = hist_encoder()
        dummy, dummy_input = external_encoder()

        fc = layers.concatenate([dummy, conv], axis=2)

        fc = layers.Conv1D(16, [7], padding='same', activation=activation)(fc)
        fc = layers.SpatialDropout1D(rate=0.3)(fc)
        fc = layers.Conv1D(8, [7], padding='same', activation=activation)(fc)
        fc = layers.SpatialDropout1D(rate=0.3)(fc)
        fc = layers.Conv1D(1, [7], padding='same')(fc)
        return keras.Model(inputs=[conv_input, dummy_input], outputs=fc), conv_input, dummy_input

    def trend_encoder():
        trend_input = keras.Input(shape=(n_steps_out, trend_length), name="full_trend")

        trend = layers.Dense(1, activation=activation)(trend_input)
        trend = layers.Dense(4, activation=activation)(trend)
        trend = layers.Conv1D(4, [5], activation=activation, padding='same')(trend)
        trend = layers.Conv1D(1, [5], activation=activation, padding='same')(trend)
        return keras.Model(inputs=[trend_input], outputs=trend), trend_input

    trend, trend_input = trend_encoder()

    fc, conv_input, dummy_input = prediction_network()

    profile_input = keras.Input(shape=(n_steps_out,), name="profile")
    profile = layers.Reshape((n_steps_out, 1))(profile_input)
    profile_model = keras.Model(inputs=[profile_input], outputs=profile)

    pred_train, pred = get_agg_layer(conv_input, dummy_input, fc, profile_input, profile_model, trend, trend_input)

    return pred_train, \
           fc, \
           trend, \
           profile_model, \
           pred


def _root_mean_squared_error(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_agg_layer(conv_input, dummy_input, fc, profile_input, profile_model, trend, trend_input):
    out = layers.concatenate([fc([conv_input, dummy_input]), profile_model(profile_input), trend(trend_input)])
    out = layers.Conv1D(1, [1], padding='same', use_bias=False, activation=activations.linear,
                        kernel_initializer=initializers.Constant(value=1 / 3), name="aggregation_layer")(out)
    pred = layers.Flatten()(out)
    model = keras.Model(inputs=[conv_input, trend_input, dummy_input, profile_input],
                        outputs=[pred])
    model.compile(optimizer=optimizers.Adam(), loss=[_sum_squared_error],
                  metrics=[_root_mean_squared_error])
    return model, keras.Model(inputs=[conv_input, trend_input, dummy_input, profile_input],
                              outputs=[pred, fc([conv_input, dummy_input]), profile_model(profile_input),
                                       trend(trend_input)])
