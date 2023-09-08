"""
This file contains the classes and functions to simulate the propagation of light.
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import dataclasses
import itertools
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import EngFormatter
from scipy import interpolate

from metabox import expansion, utils


@dataclasses.dataclass
class FieldProperties:
    """Defines the properties of a 2d field.

    Args:
        n_pixels: the number of pixels per dim after 2D expansion.
        wavelength: the wavelength of the light in meters.
        theta: the angle of the light in degrees.
        phi: the phase of the light in degrees.
        period: the period of the pixels in meters.
        upsampling: the upsampling factor.
        use_padding: whether to use padding or not.
        use_antialiasing: whether to use antialiasing or not.
    """

    n_pixels: int
    wavelength: List[float]
    theta: List[float]
    phi: List[float]
    period: float
    upsampling: int
    use_padding: bool
    use_antialiasing: bool

    def copy(self):
        """Returns a copy of the field properties."""
        return copy.deepcopy(self)


@dataclasses.dataclass
class Field2D(FieldProperties):
    """Class to store the field data and its metadata.

    Args:
        tensor: the tensor of the field.
        wavelength: the wavelength of the light in meters.
        theta: the angle of the light in degrees.
        phi: the phase of the light in degrees.
        period: the period of the pixels in meters.
        upsampling: the upsampling factor.
        use_padding: whether to use padding or not.
        use_antialiasing: whether to use antialiasing or not.
    """

    tensor: tf.Tensor

    def modulated_by(self, other: Field2D) -> Field2D:
        """Modulate this field by another field.

        Args:
            other (Field2D): the field to modulate by.

        Returns:
            Field2D: the modulated field.
        """
        # Check that the fields have the same properties.
        if not np.all(self.wavelength == other.wavelength):
            raise ValueError(
                "Wavelengths are not the same, got  {0} and {1}.".format(
                    self.wavelength, other.wavelength
                )
            )
        if not self.period == other.period:
            if self.period > other.period:
                fine_field = other.copy()
                coarse_field = self.copy()
            else:
                fine_field = self.copy()
                coarse_field = other.copy()
            enlarge_factor = coarse_field.period / float(fine_field.period)
            old_shape = coarse_field.tensor.shape[-2:]
            new_shape = (
                int(old_shape[0] * enlarge_factor),
                int(old_shape[1] * enlarge_factor),
            )
            real_part = tf.math.real(coarse_field.tensor)
            imag_part = tf.math.imag(coarse_field.tensor)

            parts = []
            for part in [real_part, imag_part]:
                part = tf.image.resize(
                    part[..., tf.newaxis],
                    new_shape,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
                part = tf.image.resize_with_crop_or_pad(
                    part,
                    fine_field.tensor.shape[-2],
                    fine_field.tensor.shape[-1],
                )
                part = part[..., 0]
                parts.append(part)
            mod_tensor = tf.complex(parts[0], parts[1])

            self = fine_field
            other = copy.deepcopy(fine_field)
            other.tensor = mod_tensor
        if not np.all(self.phi == other.phi):
            raise ValueError("angles_y are not the same.")
        if not np.all(self.theta == other.theta):
            raise ValueError("Angles are not the same.")
        if not self.tensor.shape == other.tensor.shape:
            if self.tensor.shape[-2] > other.tensor.shape[-2]:
                small_field = other
                large_field = self
            else:
                small_field = self
                large_field = other

            real_part = tf.math.real(small_field.tensor)
            imag_part = tf.math.imag(small_field.tensor)
            parts = []
            for part in [real_part, imag_part]:
                part = tf.image.resize_with_crop_or_pad(
                    part[..., tf.newaxis],
                    large_field.tensor.shape[-2],
                    large_field.tensor.shape[-1],
                )
                part = part[..., 0]
                parts.append(part)
            mod_tensor = tf.complex(parts[0], parts[1])

            self = large_field
            other = copy.deepcopy(large_field)
            other.tensor = mod_tensor

        new_field = copy.deepcopy(self)
        new_field.tensor *= other.tensor
        return new_field

    def __post_init__(self):
        # Check that the field tensor is of the correct shape.
        if len(self.tensor.shape) != 3:
            raise ValueError(
                "Field tensor must have shape [batchsize, n_pix, n_pix]"
            )

        # check batch size is correct
        n_batch = self.tensor.shape[0]
        expected_n_batch = (
            len(self.wavelength) * len(self.theta) * len(self.phi)
        )
        if n_batch != expected_n_batch:
            raise ValueError(
                """Batch size of field tensor does not match the number of
                wavelengths, angles, and angles_y multiplied together.
                Expected batch size: {0}
                Got batch size: {1}""".format(
                    expected_n_batch, n_batch
                )
            )

    def get_intensity(self):
        """Returns the intensity tensor of the tensor."""
        return get_intensity_2d(self)

    def get_phase(self):
        """Returns the phase tensor of the tensor."""
        return get_phase_2d(self)

    def wavelength_average(self):
        """Returns the wavelength averaged field."""
        return wavelength_average_2d(self)

    def show_phase(self):
        """Shows the phase of the field."""

        phase = self.get_phase()
        wl, ag = phase.shape[:2]
        diameter = self.period * phase.shape[-1]
        radius = diameter / 2.0
        for i, j in itertools.product(range(wl), range(ag)):
            wave = self.wavelength[i] * 1e6
            angle = list(itertools.product(self.theta, self.phi))[j]
            f = plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.axes([0, 0.05, 0.9, 0.9])
            im = ax.imshow(
                phase[i, j], extent=[-radius, radius, -radius, radius]
            )
            formatter0 = EngFormatter(unit="m")
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            plt.locator_params(axis="y", nbins=3)
            plt.locator_params(axis="x", nbins=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(False)
            title = "Phase Distribution\n λ={0}µm, AOI={1}°,{2}°".format(
                round(wave, 2), round(angle[0], 2), round(angle[1], 2)
            )
            ax.set_title(title)
            cax = plt.axes([0.95, 0.05, 0.05, 0.9])
            plt.colorbar(mappable=im, cax=cax)
            plt.show()

    def show_intensity(self, crop_factor=1.0):
        """Shows the intensity of the field.

        Args:
            crop_factor (float): The crop factor. Must be less than or equal to 1.0.
        """

        if crop_factor > 1.0:
            raise ValueError("Zoom must be greater than or equal to 1.0.")

        intensity = self.get_intensity()
        # put angle in the last dimension
        intensity = tf.transpose(intensity, [0, 2, 3, 1])
        if crop_factor != 1.0:
            intensity = tf.image.central_crop(intensity, crop_factor)
        # return to the original dim order
        intensity = tf.transpose(intensity, [0, 3, 1, 2])
        wl, ag = intensity.shape[:2]
        diameter = self.period * intensity.shape[-1] * crop_factor
        radius = diameter / 2.0
        for i, j in itertools.product(range(wl), range(ag)):
            wave = self.wavelength[i] * 1e6
            angle = list(itertools.product(self.theta, self.phi))[j]
            f = plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.axes([0, 0.05, 0.9, 0.9])
            im = ax.imshow(
                intensity[i, j], extent=[-radius, radius, -radius, radius]
            )
            formatter0 = EngFormatter(unit="m")
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            plt.locator_params(axis="y", nbins=3)
            plt.locator_params(axis="x", nbins=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(False)
            title = "Intensity Distribution\n λ={0}µm, AOI={1}°,{2}°".format(
                round(wave, 2), round(angle[0], 2), round(angle[1], 2)
            )
            ax.set_title(title)
            cax = plt.axes([0.95, 0.05, 0.05, 0.9])
            plt.colorbar(mappable=im, cax=cax)
            plt.show()

    def show_color_intensity(self, crop_factor=1.0):
        """Shows the intensity of the field.

        Args:
            crop_factor (float): The crop factor. Must be less than or equal to 1.0.
        """

        if crop_factor > 1.0:
            raise ValueError("Zoom must be greater than or equal to 1.0.")

        rgb_intensity = self.to_rgb_intensity()
        if crop_factor != 1.0:
            rgb_intensity = tf.image.central_crop(rgb_intensity, crop_factor)
        ag = rgb_intensity.shape[0]
        diameter = self.period * rgb_intensity.shape[-2] * crop_factor
        radius = diameter / 2.0
        for j in range(ag):
            if len(self.wavelength) == 1:
                wave = round(self.wavelength[0] * 1e6, 2)
            else:
                wave_0 = round(self.wavelength[0] * 1e6, 2)
                wave_1 = round(self.wavelength[-1] * 1e6, 2)
                wave = f"{wave_0}-{wave_1}"
            angle = list(itertools.product(self.theta, self.phi))[j]
            f = plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.axes([0, 0.05, 0.9, 0.9])
            im = ax.imshow(
                rgb_intensity[j], extent=[-radius, radius, -radius, radius]
            )
            formatter0 = EngFormatter(unit="m")
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            plt.locator_params(axis="y", nbins=3)
            plt.locator_params(axis="x", nbins=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(False)
            title = (
                "Color Intensity Distribution\n λ={0}µm, AOI={1}°,{2}°".format(
                    wave, round(angle[0], 2), round(angle[1], 2)
                )
            )
            ax.set_title(title)
            plt.show()

    def to_rgb_intensity(self):
        """Return RGB image of the intensity."""
        import colour

        intensity_distributions = self.get_intensity().numpy()
        weighted_intensity_distributions = []

        for wavelength, intensity_dist in zip(
            self.wavelength, intensity_distributions
        ):
            r, g, b = utils.wavelength_to_rgb(wavelength)
            r_dist = intensity_dist * r
            g_dist = intensity_dist * g
            b_dist = intensity_dist * b
            rgb_dist = np.stack([r_dist, g_dist, b_dist], axis=-1)
            weighted_intensity_distributions.append(rgb_dist)
        # normalize the image
        weighted_intensity_distributions = np.array(
            weighted_intensity_distributions
        )
        weighted_intensity_distributions /= np.max(
            weighted_intensity_distributions
        )
        return np.sum(weighted_intensity_distributions, axis=0)


@dataclasses.dataclass
class Field1D(FieldProperties):
    """Class to store the field data and its metadata.

    Args:
        wavelength: the wavelength of the light in meters.
        theta: the angle of the light in degrees.
        phi: the phase of the light in degrees.
        period: the period of the pixels in meters.
        upsampling: the upsampling factor.
        use_padding: whether to use padding or not.
        use_antialiasing: whether to use antialiasing or not.
    """

    tensor: tf.Tensor

    def __post_init__(self):
        # Check that the field tensor is of the correct shape.
        if len(self.tensor.shape) != 2:
            raise ValueError(
                "Field tensor must have shape [batchsize, n_pix_radius]"
            )

        # check batch size is correct
        n_batch = self.tensor.shape[0]
        expected_n_batch = (
            len(self.wavelength) * len(self.theta) * len(self.phi)
        )
        if n_batch != expected_n_batch:
            raise ValueError(
                """Batch size of field tensor does not match the number of
                wavelengths, angles, and angles_y multiplied together."""
            )

    def expand_to_2d(self, basis_dir="basis_data") -> Field2D:
        """Function to expand a 1d field to a 2d field.

        Args:
            basis_dir: the directory where the basis is saved.
                The default directory is "basis_data".

        Returns:
            propagation2d.Field2D: a 2d field
        """
        new_tensor = expansion.expand_to_2d(self.tensor, basis_dir)

        # create the 2d field
        field2d = Field2D(
            tensor=new_tensor,
            n_pixels=self.n_pixels,
            wavelength=self.wavelength,
            theta=self.theta,
            phi=self.phi,
            period=self.period,
            upsampling=self.upsampling,
            use_padding=self.use_padding,
            use_antialiasing=self.use_antialiasing,
        )

        return field2d

    def get_intensity(self):
        return get_intensity_1d(self)


def get_intensity_1d(field_1d: Field1D):
    """returns the intensity tensor of the field.

    Returns:
        tf.Tensor: intensity tensor of shape [wavelengths, angles, pixelsX]
    """

    intensity = tf.math.abs(field_1d.tensor) ** 2

    # new shape
    new_shape = tf.convert_to_tensor(
        [
            len(field_1d.wavelength),
            len(field_1d.theta) * len(field_1d.phi),
            field_1d.tensor.shape[1],
        ]
    )
    return tf.reshape(intensity, new_shape)


def get_phase_2d(field_2d: Field2D):
    """returns the phase tensor of the field.

    Returns:
        tf.Tensor: intensity tensor of shape [wavelengths, angles, pixelsX, pixelsY]
    """

    phase = tf.math.angle(field_2d.tensor)

    # new shape
    new_shape = tf.convert_to_tensor(
        [
            len(field_2d.wavelength),
            len(field_2d.theta) * len(field_2d.phi),
            field_2d.tensor.shape[1],
            field_2d.tensor.shape[2],
        ]
    )
    return tf.reshape(phase, new_shape)


def get_intensity_2d(field_2d: Field2D):
    """returns the intensity tensor of the field.

    Returns:
        tf.Tensor: intensity tensor of shape [wavelengths, angles, pixelsX, pixelsY]
    """

    intensity = tf.math.abs(field_2d.tensor) ** 2

    # new shape
    new_shape = tf.convert_to_tensor(
        [
            len(field_2d.wavelength),
            len(field_2d.theta) * len(field_2d.phi),
            field_2d.tensor.shape[1],
            field_2d.tensor.shape[2],
        ]
    )
    return tf.reshape(intensity, new_shape)


def wavelength_average_2d(field: Field2D) -> Field2D:
    """Function to average the field over the wavelengths.

    Args:
        field (Field2D): the field to average over the wavelengths.

    Returns:
        Field2D: the averaged field.
    """

    # seperate wavelength and angle dims
    field_tensor = tf.reshape(
        field.tensor,
        [
            len(field.wavelength),
            len(field.theta) * len(field.phi),
            field.n_pixels,
            field.n_pixels,
        ],
    )

    # average over the wavelengths
    new_tensor = tf.reduce_mean(field_tensor, axis=0, keepdims=False)

    # create the new field
    new_field = Field2D(
        tensor=new_tensor,
        n_pixels=field.n_pixels,
        wavelength=[np.mean(field.wavelength)],
        theta=field.theta,
        phi=field.phi,
        period=field.period,
        upsampling=field.upsampling,
        use_padding=field.use_padding,
        use_antialiasing=field.use_antialiasing,
    )

    return new_field


def get_transfer_function(
    field_like: Field2D,
    ref_idx: float,
    prop_dist: float,
    lateral_shift: Union[None, Tuple[float, float]] = None,
) -> tf.Tensor:
    """Get the Propagator object given the propagation information.

    Args:
        ref_idx (float): refractive index of the medium
        prop_dist (float): propagation distance in meters
        upsampling (int): upsampling factor
        use_padding (bool): whether or not to use padding
        use_anti_aliasing (bool): whether or not to use anti-aliasing
        lateral_shift: the lateral shift of the sampling window on the detector
            in meters. If None, the shift is set so that the Chief Ray is at the
            center of the detector. If a tuple of two floats, the shift is set
            according to the first element (x shift) and the second element (y
            shift) of the input tuple.

    Returns:
        tf.Tensor: the complex field on the final plane
    """

    return get_propagator_batched(
        ref_idx=ref_idx,
        prop_dist=prop_dist,
        n_pix=field_like.tensor.shape[1],
        period=field_like.period,
        wavelength_sampling=field_like.wavelength,
        theta_sampling=field_like.theta,
        phi_sampling=field_like.phi,
        upsampling=field_like.upsampling,
        use_padding=field_like.use_padding,
        use_antialiasing=field_like.use_antialiasing,
        lateral_shift=lateral_shift,
    )


def propagate(
    field: Field2D,
    transfer_function: tf.Tensor,
) -> Field2D:
    """Propagate a field through a given transfer function.

    Args:
        field (Field): the field to propagate
        transfer_function (tf.Tensor): the transfer function

    Returns:
        End field (Field): the complex field on the final plane
    """
    field = copy.deepcopy(field)
    new_tensor = propagate_with_propagator_batched(
        field.tensor,
        transfer_function,
        upsampling=field.upsampling,
        use_padding=field.use_padding,
    )
    field.tensor = new_tensor
    return field


def get_propagator_batched(
    ref_idx: float,
    prop_dist: float,
    n_pix: int,
    period: float,
    wavelength_sampling: List[float],
    theta_sampling: List[float],
    phi_sampling: List[float],
    upsampling=1,
    use_padding=True,
    use_antialiasing=True,
    lateral_shift: Union[None, Tuple[float, float]] = None,
) -> tf.Tensor:
    """
    Returns the transfer function for a given propagation distance.

    Args:
        ref_idx: refractive index of the medium
        prop_dist: propagation distance in meters
        n_pix: number of pixels in each dimension
        period: pixel size in meters
        wavelength: wavelength in meters
        theta_sampling: list of angles to sample in degrees
        phi_sampling: list of angles to sample in degrees
        upsampling: upsampling factor
        use_padding: whether to pad the transfer function to prevent aliasing
        use_antialiasing: whether to limit the transfer function bandwidth to
            prevent aliasing
        lateral_shift: the lateral shift of the sampling window on the detector
            in meters. If None, the shift is set so that the Chief Ray is at the
            center of the detector. If a tuple of two floats, the shift is set
            according to the first element (x shift) and the second element (y
            shift) of the input tuple.

    Returns:
        propagator: transfer function
    """
    batch_size = (
        len(wavelength_sampling) * len(theta_sampling) * len(phi_sampling)
    )

    lam0 = tf.convert_to_tensor(
        np.repeat(
            wavelength_sampling,
            np.size(theta_sampling) * np.size(phi_sampling),
        ),
        dtype=tf.float32,
    )
    lam0 = lam0[:, tf.newaxis, tf.newaxis]
    lam0 = tf.tile(lam0, multiples=(1, n_pix, n_pix))

    # Propagator definition.
    k = ref_idx * 2 * np.pi / lam0[:, 0, 0]
    k = k[:, np.newaxis, np.newaxis]
    samp = int(upsampling * n_pix)
    k = tf.cast(k, dtype=tf.complex64)

    if use_padding:
        k_xlist_pos = (
            2 * np.pi * np.linspace(0, 1 / (2 * period / upsampling), samp)
        )
        front = k_xlist_pos[-(samp - 1) :]
        front = -front[::-1]
        k_xlist = np.hstack((front, k_xlist_pos))
        k_x = np.kron(k_xlist, np.ones((2 * samp - 1, 1)))
    else:
        k_xlist = (
            2
            * np.pi
            * np.linspace(
                -1 / (2 * period / upsampling),
                1 / (2 * period / upsampling),
                samp,
            )
        )
        k_x = np.kron(k_xlist, np.ones((samp, 1)))

    k_x = k_x[np.newaxis, :, :]
    k_y = np.transpose(k_x, axes=[0, 2, 1])
    k_x = tf.convert_to_tensor(k_x, dtype=tf.complex64)
    k_x = tf.tile(k_x, multiples=(batch_size, 1, 1))
    k_y = tf.convert_to_tensor(k_y, dtype=tf.complex64)
    k_y = tf.tile(k_y, multiples=(batch_size, 1, 1))
    k_z_arg = tf.square(k) - (tf.square(k_x) + tf.square(k_y))
    k_z = tf.sqrt(k_z_arg)

    # Find shift amount
    theta_out = (
        np.pi
        / 180.0
        * tf.convert_to_tensor(
            np.tile(
                theta_sampling,
                np.size(wavelength_sampling) * np.size(phi_sampling),
            ),
            dtype=tf.float32,
        )
    )
    theta_out = theta_out[:, tf.newaxis, tf.newaxis]
    theta_out = tf.tile(theta_out, multiples=(1, n_pix, n_pix))
    theta = theta_out[:, 0, 0]
    theta = theta[:, np.newaxis, np.newaxis]
    y0_real = tf.tan(theta) * prop_dist

    phi = np.repeat(phi_sampling, np.size(theta_sampling))
    phi = np.tile(phi, np.size(wavelength_sampling))
    phi = np.pi / 180.0 * tf.convert_to_tensor(phi, dtype=tf.float32)
    phi = phi[:, tf.newaxis, tf.newaxis]
    phi = tf.tile(phi, multiples=(1, n_pix, n_pix))
    phi = phi[:, 0, 0]
    phi = phi[:, np.newaxis, np.newaxis]
    x0_real = tf.tan(phi) * prop_dist

    if lateral_shift is not None:
        x0_real = lateral_shift[0]
        y0_real = lateral_shift[1]

    y0 = tf.cast(y0_real, dtype=tf.complex64)
    x0 = tf.cast(x0_real, dtype=tf.complex64)

    propagator_arg = 1j * (k_z * prop_dist + k_x * x0 + k_y * y0)
    propagator = tf.exp(propagator_arg)

    # Limit transfer function bandwidth to prevent aliasing
    if use_antialiasing:
        S_x_src = n_pix * period
        S_y_src = S_x_src
        S_x_dest = n_pix * period
        S_y_dest = S_x_dest

        S_x_avg = 0.5 * (S_x_src + S_x_dest)
        kx_cond1 = S_x_avg < x0_real
        kx_cond2 = (-S_x_avg <= x0_real) & (x0_real < S_x_avg)

        S_y_avg = 0.5 * (S_y_src + S_y_dest)
        ky_cond1 = S_y_avg < y0_real
        ky_cond2 = (-S_y_avg <= y0_real) & (y0_real < S_y_avg)

        kx_limit_plus = k * tf.cast(
            1 / tf.math.sqrt((prop_dist / (x0_real + S_x_avg)) ** 2 + 1),
            dtype=tf.complex64,
        )
        kx_limit_minus = k * tf.cast(
            1 / tf.math.sqrt((prop_dist / (x0_real - S_x_avg)) ** 2 + 1),
            dtype=tf.complex64,
        )
        ky_limit_plus = k * tf.cast(
            1 / tf.math.sqrt((prop_dist / (y0_real + S_y_avg)) ** 2 + 1),
            dtype=tf.complex64,
        )
        ky_limit_minus = k * tf.cast(
            1 / tf.math.sqrt((prop_dist / (y0_real - S_y_avg)) ** 2 + 1),
            dtype=tf.complex64,
        )

        kx_region = tf.where(
            kx_cond1,
            _case1(k_x, k_y, k, kx_limit_minus, kx_limit_plus),
            _where_case2_case3(
                kx_cond2, k_x, k_y, k, kx_limit_minus, kx_limit_plus
            ),
        )

        ky_region = tf.where(
            ky_cond1,
            _case1(k_y, k_x, k, ky_limit_minus, ky_limit_plus),
            _where_case2_case3(
                ky_cond2, k_y, k_x, k, ky_limit_minus, ky_limit_plus
            ),
        )

        k_region = tf.cast(kx_region & ky_region, dtype=tf.complex64)

        return propagator * k_region
    else:
        return propagator


def _case1(k_x, k_y, k, kx_limit_minus, kx_limit_plus):
    """Case 1 of the transfer function bandwidth limiting."""
    kx_r1 = k_x.numpy() >= 0
    kx_r2 = (tf.square(k_x / kx_limit_minus) + tf.square(k_y / k)).numpy() >= 1
    kx_r3 = (tf.square(k_x / kx_limit_plus) + tf.square(k_y / k)).numpy() <= 1
    return kx_r1 & kx_r2 & kx_r3


def _case2(k_x, k_y, k, kx_limit_minus, kx_limit_plus):
    """Case 2 of the transfer function bandwidth limiting."""
    kx_r1 = k_x.numpy() <= 0
    kx_r2 = (tf.square(k_x / kx_limit_minus) + tf.square(k_y / k)).numpy() <= 1

    kx_r3 = k_x.numpy() > 0
    kx_r4 = (tf.square(k_x / kx_limit_plus) + tf.square(k_y / k)).numpy() <= 1
    return (kx_r1 & kx_r2) | (kx_r3 & kx_r4)


def _case3(k_x, k_y, k, kx_limit_minus, kx_limit_plus):
    """Case 3 of the transfer function bandwidth limiting."""
    kx_r1 = k_x.numpy() <= 0
    kx_r2 = (tf.square(k_x / kx_limit_plus) + tf.square(k_y / k)).numpy() >= 1
    kx_r3 = (tf.square(k_x / kx_limit_minus) + tf.square(k_y / k)).numpy() <= 1
    return kx_r1 & kx_r2 & kx_r3


def _where_case2_case3(kx_cond2, k_x, k_y, k, kx_limit_minus, kx_limit_plus):
    """Case 2 and 3 of the transfer function bandwidth limiting."""
    kx_region = tf.where(
        kx_cond2,
        _case2(k_x, k_y, k, kx_limit_minus, kx_limit_plus),
        _case3(k_x, k_y, k, kx_limit_minus, kx_limit_plus),
    )

    return kx_region


def propagate_with_propagator_batched(
    field: tf.Tensor,
    propagator: tf.Tensor,
    use_padding=True,
    upsampling=1,
) -> tf.Tensor:
    if use_padding:
        _, _, m = field.shape
        n = upsampling * m
        field = tf.transpose(field, perm=[1, 2, 0])
        field_real = tf.math.real(field)
        field_imag = tf.math.imag(field)
        field_real = tf.image.resize(field_real, [n, n], method="nearest")
        field_imag = tf.image.resize(field_imag, [n, n], method="nearest")
        field = tf.cast(field_real, dtype=tf.complex64) + 1j * tf.cast(
            field_imag, dtype=tf.complex64
        )
        field = tf.image.resize_with_crop_or_pad(field, 2 * n - 1, 2 * n - 1)
        field = tf.transpose(field, perm=[2, 0, 1])

    field_freq = tf.signal.fftshift(tf.signal.fft2d(field), axes=(1, 2))
    field_filtered = tf.signal.ifftshift(field_freq * propagator, axes=(1, 2))
    out = tf.signal.ifft2d(field_filtered)

    if use_padding:
        # Crop back down to n x n matrices
        out = tf.transpose(out, perm=[1, 2, 0])
        out = tf.image.resize_with_crop_or_pad(out, n, n)
        out = tf.transpose(out, perm=[2, 0, 1])

    return out


def propagate_with_propagator(
    field: tf.Tensor, propagator: tf.Tensor, use_padding=True, upsampling=1
) -> tf.Tensor:
    """
    Progragates the field through a distance using the transfer function.

    Args:
        field: The field to propagate.
        propagator: The transfer function.
        use_padding: Whether to use padding or not.
        upsampling: The upsampling factor.
    """
    if use_padding:
        _, m = field.shape
        field = tf.transpose(field, perm=[1, 2, 0])
        field_real = tf.math.real(field)
        field_imag = tf.math.imag(field)
        field_real = tf.image.resize(field_real, [m, m], method="nearest")
        field_imag = tf.image.resize(field_imag, [m, m], method="nearest")
        field = tf.cast(field_real, dtype=tf.complex64) + 1j * tf.cast(
            field_imag, dtype=tf.complex64
        )
        field = tf.image.resize_with_crop_or_pad(field, 2 * m - 1, 2 * m - 1)
        field = tf.transpose(field, perm=[2, 0, 1])

    field_freq = tf.signal.fftshift(tf.signal.fft2d(field))
    field_filtered = tf.signal.ifftshift(field_freq * propagator)
    out = tf.signal.ifft2d(field_filtered)

    if use_padding:
        # Crop back down to m x m matrices
        out = tf.image.resize_with_crop_or_pad(out, m, m)

    return out


def get_incident_field_2d(
    field_props: FieldProperties,
) -> tf.Tensor:
    """Defines the input electric fields for the given wavelengths and field angles.

    Args:
        field_props: the field properties.

    Returns:
        Field2d: The incident field.
    """

    # Define the cartesian cross section
    # TODO: perioidicity for x and y seperately
    wavelength_base = field_props.wavelength
    theta_base = field_props.theta
    phi_base = field_props.phi

    # Define the cartesian cross section
    n_pix = field_props.n_pixels
    dx = field_props.period
    dy = field_props.period
    xa = np.linspace(0, n_pix - 1, n_pix) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.linspace(0, n_pix - 1, n_pix) * dy  # y axis vector
    ya = ya - np.mean(ya)  # center y axis at zero
    [y_mesh, x_mesh] = np.meshgrid(ya, xa)
    x_mesh = x_mesh[np.newaxis, :, :]
    y_mesh = y_mesh[np.newaxis, :, :]

    lam0, theta, phi = utils.unravel_wavelength_theta_phi(
        wavelength_base, theta_base, phi_base
    )
    lam0 = lam0[:, tf.newaxis, tf.newaxis]
    theta = theta[:, tf.newaxis, tf.newaxis]
    phi = phi[:, tf.newaxis, tf.newaxis]

    phase_def = (
        2 * np.pi / lam0 * (np.sin(theta) * x_mesh + np.sin(phi) * y_mesh)
    )
    phase_def = tf.cast(phase_def, dtype=tf.complex64)

    tensor = tf.exp(1j * phase_def)
    total_energy = tf.reduce_sum(tf.math.abs(tensor) ** 2)
    tensor /= tf.sqrt(tf.cast(total_energy, dtype=tf.complex64))

    return Field2D(
        tensor=tensor,
        n_pixels=field_props.n_pixels,
        wavelength=field_props.wavelength,
        theta=field_props.theta,
        phi=field_props.phi,
        period=field_props.period,
        upsampling=field_props.upsampling,
        use_padding=field_props.use_padding,
        use_antialiasing=field_props.use_antialiasing,
    )
