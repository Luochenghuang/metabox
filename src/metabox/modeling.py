from metabox import rcwa, utils
import itertools, dataclasses, os, pickle, json, copy
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union, Dict, Any
from tqdm.keras import TqdmCallback


@dataclasses.dataclass
class SimulationLibrary:
    protocell: rcwa.ProtoUnitCell
    incidence: rcwa.Incidence
    sim_config: dict
    feature_values: np.ndarray
    simulation_output: tf.Tensor

    """Stores the simulation parameters and the simulation output.

    Attributes:
        protocell: the simulated protocell.
        incidence: the incidence of the light.
        sim_config: the simulation configuration.
        feature_values: the sampled feature values.
        simulation_output: the simulation output.
    """

    def get_training_x(self) -> np.ndarray:
        """Returns the training input.

        First dim is the wavelength, the second dim is the first parameterized
        feature, the third dim is the second parameterized feature, etc.
        """
        # check SimulationLibrary to make sure the wavelength is the only varying
        # parameter in `Incidence`
        if len(self.incidence.wavelength) != self.simulation_output.shape[0]:
            raise ValueError(
                "The number of wavelengths in the simulation output doesn't match "
                "the number of wavelengths in the incidence."
                "Training doesn't support multiple incident angles at the moment."
            )

        # assign names to the useful data
        feature_values = self.feature_values
        wavelength_values = self.incidence.wavelength
        output_values = self.simulation_output

        # find various of shapes
        n_wavelengths, n_instances, _ = output_values.shape
        n_features = feature_values.shape[0]

        # expand the feature values to include the wavelength
        # TODO: use numpy operations instead of tf operations
        wavelengths = tf.cast(wavelength_values, tf.float32)[:, tf.newaxis]
        wavelengths = tf.tile(wavelengths, [1, n_instances])
        wavelengths = wavelengths[tf.newaxis, ...]
        expanded_features = feature_values[:, tf.newaxis, ...]
        expanded_features = tf.cast(expanded_features, tf.float32)
        expanded_features = tf.tile(expanded_features, [1, n_wavelengths, 1])
        expanded_features = tf.concat([wavelengths, expanded_features], axis=0)
        expanded_features = tf.reshape(expanded_features, [n_features + 1, -1])
        return expanded_features.numpy().T

    def get_training_y(self) -> np.ndarray:
        """Returns the training output."""
        output_values = self.simulation_output
        return output_values.reshape(-1, 2)

    def save(self, name: str, path: str, overwrite: bool = False):
        """Saves the simulation library

        Args:
            name (str): the name of the library
            path (str): a path to save the library to
            overwrite (bool): Whether or not to overwrite exisiting library.
        """
        save_simulation_library(
            self,
            name=name,
            path=path,
            overwrite=overwrite,
        )


def sample_protocell(
    protocell,
    incidence,
    sim_config,
) -> None:
    """Sample a protocell with a given incidence.

    For a given protocell, each unique `Feature` is sampled given its sampling
        number (see `Feature.sampling`). Then the permutation of the sampled
        features is simulated with the given incidence. The output is a

    Args:
        protocell (Protocell): The protocell to sample.
        incidence (float): The incidence to sample the protocell with.
        sim_config (dict): The simulation configuration.

    Returns:
        SimulationLibrary: The simulation library.
    """

    features = protocell.features
    sampling_values_per_feature = []
    for feature in features:
        if feature.sampling is None:
            raise ValueError(f"Feature {feature.name} has no sampling value.")
        uniform_sampling = np.linspace(
            feature.vmin, feature.vmax, feature.sampling
        )
        sampling_values_per_feature.append(uniform_sampling)
    # create permutations of the sampling values
    feature_values = list(itertools.product(*sampling_values_per_feature))
    feature_values = np.array(feature_values).T

    output = rcwa.simulate_parameterized_unit_cells(
        feature_values,
        protocell,
        incidence,
        sim_config,
    ).numpy()

    return SimulationLibrary(
        protocell=protocell,
        incidence=incidence,
        sim_config=sim_config,
        feature_values=feature_values,
        simulation_output=output,
    )


@tf.keras.utils.register_keras_serializable(package="modeling")
class NormComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a complex feature column."""

    def __init__(self, **kwargs):
        super(NormComplexLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # split into real and imag
        a, b = tf.split(inputs, 4, axis=-1)
        # combines two features into a complex feature
        norm = tf.sqrt(a**2 + b**2)
        norm = tf.clip_by_value(norm, 1, np.Inf)
        a = tf.cast(a / norm, tf.complex64)
        b = tf.cast(b / norm, tf.complex64)
        return a + b * 1j


@tf.keras.utils.register_keras_serializable(package="modeling")
class ComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a complex feature column."""

    def __init__(self, **kwargs):
        super(ComplexLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # split into real and imag
        a, b = tf.split(inputs, 2, axis=-1)
        # combines two features into a complex feature
        a = tf.cast(a, tf.complex64)
        b = tf.cast(b, tf.complex64)
        return a + b * 1j


@tf.keras.utils.register_keras_serializable(package="modeling")
def euclidian_distance(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Calculates the euclidian distance between two complex numbers.

    Args:
        y_true (tf.Tensor): true value.
        y_pred (tf.Tensor): predicted value.

    Returns:
        tf.Tensor: the euclidian distance between the two complex numbers.
    """
    return tf.abs(y_true - y_pred) ** 2


def create_fcc_model(
    normalizer: tf.keras.layers.Normalization,
    optimizer: tf.keras.optimizers.Optimizer,
    hidden_layer_units_list: List[int],
    activation_list: List[str],
    limit_output_to_unity: bool = False,
) -> tf.keras.Sequential:
    """Creates a simple fully connected network with a normalization layer.

    Args:
        normalizer (tf.keras.layers.Normalization): the normalization layer.
        optimizer (tf.keras.optimizers.Optimizer): the optimizer.
        hidden_layer_units (List[int]): the number of units in each hidden layer.
        activation (List[str]): the activation function for each hidden layer.
        limit_output_to_unity (bool): whether to limit the intensity to unity.
    """
    # Defines the model
    layers = [normalizer]
    for n_units, activation in zip(hidden_layer_units_list, activation_list):
        layers.append(tf.keras.layers.Dense(n_units, activation=activation))
    layers.append(tf.keras.layers.Dense(4))
    if limit_output_to_unity:
        layers.append(NormComplexLayer())
    else:
        layers.append(ComplexLayer())

    model = tf.keras.Sequential(layers)
    optimizer.build(model.trainable_variables)
    model.compile(loss=euclidian_distance, optimizer=optimizer)
    return model


@dataclasses.dataclass
class Metamodel:
    model: tf.keras.Sequential
    history: tf.keras.callbacks.History
    protocell: rcwa.ProtoUnitCell
    sim_config: dict

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
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        model_path = os.path.join(new_folder, "tf_model")
        pkl_path = os.path.join(new_folder, name + ".pkl")
        json_path = os.path.join(new_folder, "info.json")
        self.model.save(model_path.format(0), overwrite=overwrite)

        if not overwrite:
            if os.path.exists(pkl_path):
                raise ValueError("File already exists.")

        filehandler_pkl = open(pkl_path, "wb")
        with utils.suppress_stdout_stderr():
            new_self = copy.deepcopy(self)
            del new_self.model
            pickle.dump(new_self, filehandler_pkl)
        json_dict = dataclasses.asdict(self.protocell)
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

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
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
        # TODO: make compatible with protocell
        for feature in self.protocell.features:
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
        my_model.model = tf.keras.models.load_model(model_path)
    return my_model


def create_and_train_model(
    sim_lib: SimulationLibrary,
    n_epochs: int = 100,
    optimizer: tf.keras.optimizers.Optimizer = None,
    hidden_layer_units_list: List[int] = [64, 128, 256, 64],
    activation_list: List[str] = ["relu", "relu", "relu", "relu"],
    limit_output_to_unity: bool = False,
    train_batch_size: Union[None, int] = None,
    validation_split: float = 0.05,
    verbose: int = 0,
) -> Tuple[tf.keras.Sequential, tf.keras.callbacks.History]:
    """Creates and fits a given model to the atom library.

    Fits a fully connected network to the atom library. The network is
    a simple fully connected network with a normalization layer.
    Returns the trained model and the history of the training.

    Args:
        sim_lib (SimulationLibrary): the simulation library.
        n_epochs (int): the number of epochs.
        optimizer (tf.keras.optimizers.Optimizer): the optimizer.
        hidden_layer_units (List[int]): the number of units in each hidden layer.
        activation (List[str]): the activation function for each hidden layer.
        limit_output_to_unity (bool): whether to limit the intensity to unity.
        train_batch_size (Union[None, int]): the batch size for training.
        validation_split (float): the fraction of the training data to use for
            validation.
        verbose (int): the verbosity level.

    Returns:
        MetaModel: Contains the trained model, the history of the training,
            and the proto-atom used to train the model.
    """
    if len(hidden_layer_units_list) != len(activation_list):
        raise ValueError(
            "The number of hidden layers must be equal to the number of activation functions."
        )
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    train_input = sim_lib.get_training_x()
    train_output = sim_lib.get_training_y()

    normalizer = tf.keras.layers.Normalization(
        axis=-1, input_dim=train_input.shape[-1]
    )
    normalizer.adapt(train_input)

    model = create_fcc_model(
        normalizer=normalizer,
        optimizer=optimizer,
        hidden_layer_units_list=hidden_layer_units_list,
        activation_list=activation_list,
        limit_output_to_unity=limit_output_to_unity,
    )

    # Train the model seperately
    history = model.fit(
        train_input.astype(np.float32),
        train_output.astype(np.complex64),
        validation_split=validation_split,
        verbose=verbose,
        epochs=n_epochs,
        batch_size=train_batch_size,
        callbacks=[TqdmCallback(verbose=1)],
    )

    return Metamodel(
        model=model,
        history=history,
        protocell=sim_lib.protocell,
        sim_config=sim_lib.sim_config,
    )


def save_simulation_library(
    sim_lib: SimulationLibrary,
    name: str,
    path: str,
    overwrite: bool = False,
) -> None:
    """Saves the metamodel to a file.

    Args:
        filename (str): The name of the file to save the metamodel to.
        path (str): The path to the file.
    """
    if not os.path.isdir(path):
        raise ValueError("path must be a directory.")
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, name + ".pkl")
    if not overwrite:
        if os.path.exists(full_path):
            raise ValueError("File already exists.")

    filehandler_pkl = open(full_path, "wb")
    pickle.dump(sim_lib, filehandler_pkl)
    json_dict = dataclasses.asdict(sim_lib)
    for item in json_dict:
        json_dict[item] = utils.recursively_convert_ndarray_in_dict_to_list(
            item
        )
        if np.any(np.iscomplex(json_dict[item])):
            json_dict[item] = str(json_dict[item])
    json_d = json.dumps(json_dict, indent=4)
    with open(full_path + ".json", "w") as outfile:
        outfile.write(json_d)

    print("Saved the atom library to " + full_path)


def load_simulation_library(
    name: str,
    path: str,
) -> SimulationLibrary:
    """Loads a AtomLibrary from a file.

    Args:
        name (str): The name of the file to load the atom library from.
        path (str): The path to the file.

    Returns:
        AtomLibrary: The loaded atom library.
    """
    full_path = os.path.join(path, name + ".pkl")
    if not os.path.exists(full_path):
        raise ValueError("File does not exist.")
    filehandler = open(full_path, "rb")
    return pickle.load(filehandler)
