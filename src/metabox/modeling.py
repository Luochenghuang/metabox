"""
This module defines classes and functions to create metamodel for proto-atoms[0].

[0] A `proto-atom` is a meta-atom that has one or more `Feature`s[1].
[1] A `Feature` is a variable in the simulation definition of a meta-atom.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import dataclasses
import json
import pickle
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback

from metabox import atom, utils


@dataclasses.dataclass
class Metamodel:
    trained_model: keras.Sequential
    training_history: keras.callbacks.History
    proto_atom: atom.Atom

    def save(
        self,
        name: str,
        path: str = "./saved_metamodels",
        overwrite: bool = False,
    ) -> None:
        """Saves the metamodel to a file.

        Args:
            filename (str): The name of the file to save the metamodel to.
            path (str): The path to the file.
        """
        new_folder = os.path.join(path, name)
        if not os.path.isdir(path):
            raise ValueError("path must be a directory.")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        model_path = os.path.join(new_folder, "tf_model")
        pkl_path = os.path.join(new_folder, name + ".pkl")
        json_path = os.path.join(new_folder, "info.json")
        self.trained_model.save(model_path, overwrite=overwrite)

        if not overwrite:
            if os.path.exists(pkl_path):
                raise ValueError("File already exists.")

        filehandler_pkl = open(pkl_path, "wb")
        with utils.suppress_stdout_stderr():
            new_self = copy.deepcopy(self)
            del new_self.trained_model
            pickle.dump(new_self, filehandler_pkl)
        json_dict = self.proto_atom.to_dict()
        for item in json_dict:
            if np.iscomplex(json_dict[item]):
                json_dict[item] = str(json_dict[item])
        json_d = json.dumps(json_dict, indent=4)
        with open(json_path, "w") as outfile:
            outfile.write(json_d)
        print("Saved metamodel to " + new_folder)

    def plot_training_history(self) -> None:
        """Plots the training history."""
        import matplotlib.pyplot as plt

        plt.plot(self.training_history.history["loss"])
        plt.plot(self.training_history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def set_feature_constraint(
        self,
        feature_str: str,
        vmin: Union[float, None],
        vmax: Union[float, None],
    ) -> None:
        """Sets the value of a feature.

        Args:
            feature (str): The attribute name of the feature to set.
            vmin: The minimum value of the feature.
            vmax: The maximum value of the feature.
        """
        for feature in self.proto_atom.unique_features:
            if feature_str == feature.name:
                break

        current_vmin = feature.vmin
        current_vmax = feature.vmax

        if vmin is None:
            vmin = current_vmin
        if vmax is None:
            vmax = current_vmax
        if vmin < current_vmin:
            raise ValueError(
                "Minimum constraint must be greater or equal to the minimum feature value."
            )
        if vmax > current_vmax:
            raise ValueError(
                "Maximum constraint must be smaller or equal to the maximum feature value."
            )
        setattr(feature, "vmin", vmin)
        setattr(feature, "vmax", vmax)


def load_metamodel(
    name: str,
    save_dir: str = "./saved_metamodels",
) -> Metamodel:
    """Loads a metamodel from a file.

    Args:
        name (str): The name of the file to load the metamodel from.
        path (str): The path to the file.

    Returns:
        MetaModel: The loaded metamodel.
    """
    new_folder = os.path.join(save_dir, name)
    model_path = os.path.join(new_folder, "tf_model")
    pkl_path = os.path.join(new_folder, name + ".pkl")
    if not os.path.exists(pkl_path):
        raise ValueError("File does not exist.")
    filehandler = open(pkl_path, "rb")
    with utils.suppress_stdout_stderr():
        my_model = pickle.load(filehandler)
        my_model.trained_model = keras.models.load_model(model_path)
    return my_model


def fit_atom_lib_fcc(
    atom_lib: atom.AtomLibrary,
    n_layers: int = 4,
    n_units: int = 256,
    n_epochs: int = 100,
    train_batch_size: Union[None, int] = None,
    learning_rate: float = 0.001,
    limit_output_to_unity: bool = True,
    activation: str = "relu",
    validation_split: float = 0.05,
    verbose: int = 0,
) -> Tuple[tf.keras.Sequential, keras.callbacks.History]:
    """Fits a fully connected network to the atom library.

    Fits a fully connected network to the atom library. The network is
    a simple fully connected network with a normalization layer.
    Returns the trained model and the history of the training.

    Args:
        atom_lib (atom.AtomLibrary): The atom library to fit.
        n_layers (int, optional): the number of fcc layers. Defaults to 4.
        n_units (int, optional): the number of units per layer. Defaults to 256.
        n_epochs (int, optional): the number of epochs. Defaults to 100.
        train_batch_size (Union[None, int], optional): the batch size for training.
            Defaults to None.
        learning_rate (float, optional): the learning rate. Defaults to 0.001.
        limit_output_to_unity (bool, optional): whether to limit the amplitude to unity.
        activation (str, optional): the activation function. Defaults to "relu".
        validation_split (float, optional): the validation split. Defaults to 0.05.
        verbose (int, optional): verbosity. Defaults to 0.

    Returns:
        MetaModel: Contains the trained model, the history of the training,
            and the proto-atom used to train the model.
    """

    input_features, output = atom_lib.get_training_data()
    normalizer = tf.keras.layers.Normalization(
        axis=-1, input_dim=input_features.shape[-1]
    )
    normalizer.adapt(input_features)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model = simple_fcc(
        normalizer,
        optimizer,
        n_layers=n_layers,
        n_units=n_units,
        activation=activation,
        limit_output_to_unity=limit_output_to_unity,
    )

    history = model.fit(
        input_features,
        output.astype(np.complex128),
        validation_split=validation_split,
        verbose=verbose,
        epochs=n_epochs,
        batch_size=train_batch_size,
        callbacks=[TqdmCallback(verbose=1)],
    )

    return Metamodel(model, history, atom_lib.proto_atom)


@keras.utils.register_keras_serializable(package="modeling")
class ComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a complex feature column."""

    def __init__(self, **kwargs):
        super(ComplexLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # split into real and imag
        a, b = tf.split(inputs, 2, axis=-1)
        a = tf.cast(a, tf.complex128)
        b = tf.cast(b, tf.complex128)
        return a + b * 1j


@keras.utils.register_keras_serializable(package="modeling")
class NormComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a complex feature column."""

    def __init__(self, **kwargs):
        super(NormComplexLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # split into real and imag
        a, b = tf.split(inputs, 2, axis=-1)
        norm = tf.sqrt(a**2 + b**2)
        norm = tf.clip_by_value(norm, 1, np.Inf)
        a = tf.cast(a / norm, tf.complex128)
        b = tf.cast(b / norm, tf.complex128)
        return a + b * 1j


@keras.utils.register_keras_serializable(package="modeling")
def euclidian_distance(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Calculates the euclidian distance between two complex numbers.

    Args:
        y_true (tf.Tensor): true value.
        y_pred (tf.Tensor): predicted value.

    Returns:
        tf.Tensor: the euclidian distance between the two complex numbers.
    """
    return tf.abs(y_true - y_pred) ** 2


def simple_fcc(
    normalizer: tf.keras.layers.Normalization,
    optimizer,
    n_layers: int = 2,
    n_units: int = 64,
    activation: str = "relu",
    limit_output_to_unity: bool = False,
) -> tf.keras.Sequential:
    """Creates a simple fully connected network with a normalization layer."""
    # Defines the model
    layers = [normalizer]
    for _ in tf.range(n_layers):
        layers.append(keras.layers.Dense(n_units, activation=activation))
    layers.append(keras.layers.Dense(2))
    if limit_output_to_unity:
        layers.append(NormComplexLayer())
    else:
        layers.append(ComplexLayer())

    model = keras.Sequential(layers)
    model.compile(loss=euclidian_distance, optimizer=optimizer)
    return model
