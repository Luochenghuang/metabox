import dataclasses
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf


# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
@contextmanager
def suppress_stdout_stderr() -> None:
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def recursively_convert_ndarray_in_dict_to_list(item: Any):
    """Recursively converts ndarray item in dict to list"""
    if type(item) is np.ndarray:
        item = item.tolist()
    elif type(item) is dict:
        for key, value in item.items():
            item[key] = recursively_convert_ndarray_in_dict_to_list(value)
    return item


@dataclasses.dataclass
class Feature:
    """Defines a feature variable.

    Args:
        vmin: the minimum value of the feature.
        vmax: the maximum value of the feature.
        name: the name of the feature.
        sampling: the number of samples to take between vmin and vmax. If None,
            the sampling is undefined.
    """

    vmin: float
    vmax: float
    name: str
    initial_value: Union[float, None] = None
    sampling: Union[int, None] = None
    value: Union[tf.Variable, None, float] = None

    def _post_init__(self):
        if ":" in self.name:
            raise ValueError(
                "The name of the feature cannot contain the character '~'"
            )

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def initialize_value(self) -> None:
        """Initializes the variables of the feature."""
        if self.initial_value is not None:
            tensor = tf.math.real(self.initial_value)
            tensor = tf.cast([tensor], tf.float32)
        else:
            tensor = tf.random.uniform([1], self.vmin, self.vmax, tf.float32)
        self.value = tensor

    def set_variable(self) -> None:
        """Convert self.value to a variable"""
        constraint_func = lambda x: tf.clip_by_value(x, self.vmin, self.vmax)
        self.value = tf.Variable(
            self.value, constraint=constraint_func, name=self.name
        )

    def set_value(self, value: Any) -> None:
        """Set the value of the feature to value"""
        self.value = value


@dataclasses.dataclass
class Incidence:
    """Defines the physical properties of the incident light.

    Args:
        wavelength: the wavelengths of the light in meters.
        theta: tuple of the angles of incidence in degrees on the xz plane.
            Defaults to (0).
        phi: tuple of the angles of incidence in degrees on the yz plane.
            Defaults to (0).
        jones_vector: the Jones vector of the incident light.
            Defaults to (1, 0) which corresponds to a linearly polarized
            light with the electric field vector parallel to the x axis.
    """

    wavelength: Tuple[float]
    theta: Tuple[float] = (0,)
    phi: Tuple[float] = (0,)
    jones_vector: Tuple[float] = (1, 0)


def unravel_wavelength_theta_phi(
    wavelength: List[float], theta: List[float], phi: List[float]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Unravels the wavelength, theta, and phi lists into tensors.

    Args:
        wavelength: a list of wavelengths in meters.
        theta: a list of angles of incidence in degrees on the xz plane.
        phi: a list of angles of incidence in degrees on the yz plane.

    Returns:
        A tuple of tensors of shape (batch_size,).
    """
    wavelength_base = wavelength
    theta_base = theta
    phi_base = phi

    wavelength_out = tf.convert_to_tensor(
        np.repeat(wavelength_base, np.size(theta_base) * np.size(phi_base)),
        dtype=tf.float32,
    )

    theta_out = (
        np.pi
        / 180.0
        * tf.convert_to_tensor(
            np.tile(theta_base, np.size(wavelength_base) * np.size(phi_base)),
            dtype=tf.float32,
        )
    )

    phi = np.repeat(phi_base, np.size(theta_base))
    phi = np.tile(phi, np.size(wavelength_base))
    phi_out = np.pi / 180.0 * tf.convert_to_tensor(phi, dtype=tf.float32)

    return wavelength_out, theta_out, phi_out


def unravel_incidence(incidence: Incidence) -> Dict[str, Any]:
    """Serializes an incidence data into lists."""
    (
        wavelength_batch,
        theta_batch,
        phi_batch,
    ) = unravel_wavelength_theta_phi(
        wavelength=incidence.wavelength,
        theta=incidence.theta,
        phi=incidence.phi,
    )
    x_pol, y_pol = incidence.jones_vector
    x_pol_batch = tf.cast(
        tf.repeat(x_pol, len(wavelength_batch)), tf.complex64
    )
    y_pol_batch = tf.cast(
        tf.repeat(y_pol, len(wavelength_batch)), tf.complex64
    )

    return {
        "wavelength": wavelength_batch,
        "theta": theta_batch,
        "phi": phi_batch,
        "ptm": x_pol_batch,
        "pte": y_pol_batch,
    }


ParameterType = Union[Feature, float, tf.Tensor]
CoordType = Tuple[ParameterType, ParameterType]

TF_FUNCTIONS = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "exp",
    "sqrt",
    "square",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "matmul",
    "transpose",
    "reshape",
    "expand_dims",
    "squeeze",
    "stack",
    "concat",
    # Add any other Tensor functions you want to support here
]


def wavelength_to_rgb(wavelength):
    """
    Convert a wavelength in the visible spectrum to RGB
    Input: wavelength (in m)
    Output: RGB tuple
    """
    wavelength = wavelength * 1e9
    gamma = 0.8

    if (wavelength >= 380) and (wavelength < 440):
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif (wavelength >= 440) and (wavelength < 490):
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif (wavelength >= 510) and (wavelength < 580):
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return (R, G, B)
