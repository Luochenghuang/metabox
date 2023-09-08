"""
Defines a lens assembly and functionalities for simualting the performances of the lens assembly.
"""
import copy, os, dataclasses, enum, itertools, logging, dill, tqdm, re
from PIL import Image
from typing import List, Tuple, Union, Dict

import numpy as np
import tensorflow as tf
from matplotlib.ticker import EngFormatter

from metabox import (
    expansion,
    metrics,
    modeling,
    propagation,
    rcwa,
    utils,
)
from metabox.utils import Incidence

# Suppress tensorflow warnings
tf.get_logger().setLevel(logging.ERROR)


@dataclasses.dataclass
class AtomArray2D:
    """Class to store the 2D atom array data and its metadata.

    Args:
        tensor: the atom structure array tensor with shape (n_features, n_atoms)
        period: the period of the atom array in meters.
        mmodel: the `MetaModel` used to generate the atom array.
            The `MetaModel` stores the trained model and the structure of the atom.
        proto_unit_cell: the proto unit cell (rcwa.ProtoUnitCell)
        cached_fields: the cached transmission coefficients for the atom array.
    """

    tensor: tf.Tensor
    period: float
    mmodel: modeling.Metamodel = None
    proto_unit_cell: rcwa.ProtoUnitCell = None
    cached_fields: List[tf.Tensor] = None

    def __post_init__(self):
        has_mmodel = self.mmodel is not None
        has_unit_cell = self.proto_unit_cell is not None
        if has_mmodel and has_unit_cell:
            raise ValueError(
                "Cannot have both a mmodel and a parameterized unit cell."
            )
        if not has_mmodel and not has_unit_cell:
            raise ValueError(
                "Must have either a mmodel or a parameterized unit cell."
            )
        self.use_mmodel = has_mmodel

    def find_feature_index_excluding_wavelength(self, feature_str: str):
        """Returns the index of the feature in the structure tensor.

        Args:
            feature_str: the name of the feature.

        Raises:
            ValueError: if the feature is not found in the atom array.
        """
        all_features = copy.deepcopy(self.mmodel.protocell.features)
        if not feature_str in [f.name for f in all_features]:
            raise ValueError(
                "Feature {} not found in the atom array.".format(feature_str)
            )
        index = 0
        for i in range(len(all_features)):
            if all_features[i].name == feature_str:
                break
            index += 1
        return index

    def get_atom_array(self, incidence: "Incidence") -> List[rcwa.UnitCell]:
        return AtomArray1D(
            self.tensor,
            self.period,
            self.mmodel,
        ).get_atom_array(incidence)

    def get_feature_map(self, feature: str) -> tf.Tensor:
        """Returns the structure of the atom array.

        Args:
            feature: the feature string to get the structure of.

        Returns:
            tf.Tensor: the structure of the atom array.
        """
        if self.use_mmodel:
            index = self.find_feature_index_excluding_wavelength(feature)
        else:
            index = self.proto_unit_cell.get_feature_index(feature)
        matrix_width = int(np.sqrt(self.tensor.shape[-1]))
        return tf.reshape(self.tensor[index], [matrix_width, matrix_width])

    def set_feature_map(
        self, feature: str, new_values: Union[np.ndarray, tf.Tensor]
    ) -> tf.Tensor:
        """Change the structure feature of the atom array to the given values.

        Args:
            feature: the feature to get the structure of.

        Returns:
            tf.Tensor: the structure of the atom array.
        """

        if isinstance(self.tensor, tf.Variable):
            raise NotImplementedError(
                "Changing the feature map of a tf.Variable is not implemented yet."
            )

        index = self.find_feature_index_excluding_wavelength(feature)
        matrix_width = int(np.sqrt(self.tensor.shape[-1]))

        if np.shape(new_values) != (matrix_width, matrix_width):
            raise ValueError(
                "The new values must have the same shape as the feature map ({0}, {0}).".format(
                    matrix_width
                )
            )

        tsnp = self.tensor.numpy()
        tsnp[index] = new_values.flatten()
        self.tensor = tf.convert_to_tensor(tsnp)

    def show_feature_map(self, only_feature: Union[str, None] = None):
        """Shows the structure of the atom array.

        Args:
            only_feature: the only feature to show the structure of if not None.
                Shows all features if None.
        """

        n_pixels = int(np.sqrt(self.tensor.shape[-1]))
        diameter = self.period * n_pixels
        radius = diameter / 2.0
        features_with_wavelength = copy.deepcopy(
            self.mmodel.protocell.features
        )
        features_with_wavelength = [
            feature.name for feature in features_with_wavelength
        ]
        all_features = []
        for feature_str in features_with_wavelength:
            all_features.append(feature_str)

        if only_feature is not None:
            if not only_feature in features_with_wavelength:
                raise ValueError(
                    "Feature {} not found in the atom array.".format(
                        only_feature
                    )
                )
            all_features = [only_feature]

        for i in range(len(all_features)):
            feature_array = self.get_feature_map(all_features[i])
            f = plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.axes([0, 0.05, 0.9, 0.9])
            im = ax.imshow(
                feature_array, extent=[-radius, radius, -radius, radius]
            )
            formatter0 = EngFormatter(unit="m")
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            plt.locator_params(axis="y", nbins=3)
            plt.locator_params(axis="x", nbins=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(False)
            title = "Feature: {}".format(all_features[i])
            ax.set_title(title)
            cax = plt.axes([0.95, 0.05, 0.05, 0.9])
            plt.colorbar(mappable=im, cax=cax)
            plt.show()

    def set_to_use_rcwa(self):
        """Skips the metamodel and directly simulate the atom array using RCWA.

        Note that this method will change the atom array permanently.
        This method is useful for verifying the performance of the metamodel.
        """
        if not self.use_mmodel:
            print("The atom array is already using RCWA simulation directly.")
            return
        metamodel = self.mmodel
        protocell = metamodel.protocell
        self.proto_unit_cell = protocell
        self.use_mmodel = False
        self.cached_fields = None
        self.sim_config = metamodel.sim_config


@dataclasses.dataclass
class AtomArray1D:
    """Class to store the 1D atom array data and its metadata.

    Args:
        tensor: the atom structure array tensor with shape (n_features, n_atoms)
        period: the period of the atom array in meters.
        mmodel: the `MetaModel` used to generate the atom array.
            The `MetaModel` stores the trained model and the structure of the atom.
        proto_unit_cell: the proto unit cell (rcwa.ProtoUnitCell)
    """

    tensor: tf.Tensor
    period: float
    mmodel: modeling.Metamodel = None
    proto_unit_cell: rcwa.ProtoUnitCell = None

    def __post_init__(self):
        has_mmodel = self.mmodel is not None
        has_unit_cell = self.proto_unit_cell is not None
        if has_mmodel and has_unit_cell:
            raise ValueError(
                "Cannot have both a mmodel and a parameterized unit cell."
            )
        if not has_mmodel and not has_unit_cell:
            raise ValueError(
                "Must have either a mmodel or a parameterized unit cell."
            )
        self.use_mmodel = has_mmodel
        self.cached_fields = None

    def find_feature_index_excluding_wavelength(self, feature_str: str):
        """Returns the index of the feature in the structure tensor.

        Args:
            feature_str: the name of the feature.

        Raises:
            ValueError: if the feature is not found in the atom array.
        """
        all_features = copy.deepcopy(self.mmodel.protocell.features)
        if not feature_str in [f.name for f in all_features]:
            raise ValueError(
                "Feature {} not found in the atom array.".format(feature_str)
            )
        index = 0
        for i in range(len(all_features)):
            if all_features[i].name == feature_str:
                break
            index += 1
        return index

    def expand_to_2d(self, basis_dir="basis_data") -> AtomArray2D:
        """Function to expand a 1d atom array to a 2d atom array.

              Args:
                  basis_dir: the directory where the basis is saved.
        "vim.normalModeKeyBindingsNonRecursive": [
                      The default directory is "basis_data".

              Attributes:
                  tensor: the atom structure array tensor.
                      The outmost dimension is the feature dimension.

              Returns:
                  AtomArray2D: a 2d atom array
        """
        new_tensor = expansion.expand_to_2d(self.tensor, basis_dir)
        # clapse the last two dimensions
        new_shape = list(new_tensor.shape)
        new_shape = new_shape[:-2] + [-1]
        new_tensor = tf.reshape(new_tensor, new_shape)

        return AtomArray2D(new_tensor, self.period, self.mmodel)

    def get_atom_array(self, incidence: "Incidence") -> List[rcwa.UnitCell]:
        """Returns the batched atom array with shape (n_batch, n_atoms)."""
        if self.use_mmodel:
            return self.mmodel.protocell.generate_cells_from_parameter_tensor(
                self.tensor
            )
        return self.proto_unit_cell.generate_cells_from_parameter_tensor(
            self.tensor
        )

    def get_feature_map(self, feature: str) -> np.ndarray:
        """Returns the 2D feature array.

        Args:
            feature: the feature to return the array of.

        Returns:
            np.ndarray: the feature array.
        """
        self.expand_to_2d().get_feature_map(feature)

    def get_feature_map_1d(self, feature_str: str) -> np.ndarray:
        """Returns the 1D feature array.

        Args:
            feature_str: the feature string to return the array of.

        Returns:
            np.ndarray: the feature array.
        """
        if self.use_mmodel:
            index = self.find_feature_index_excluding_wavelength(feature_str)
        else:
            index = self.proto_unit_cell.find_feature_index(feature_str)
        return self.tensor[index, :].numpy()

    def set_feature_map(self, feature: str, feature_array: np.ndarray):
        """Sets the 2D feature array.

        Args:
            feature: the feature to set the array of.
            feature_array: the feature array to set.
        """
        if self.use_mmodel:
            index = self.find_feature_index_excluding_wavelength(feature)
        else:
            index = self.proto_unit_cell.find_feature_index(feature)
        tsnp = self.tensor.numpy()
        tsnp[index, :] = feature_array
        self.tensor = tf.convert_to_tensor(tsnp)

    def show_features(self, only_feature: Union[str, None] = None):
        """Shows the structure of the atom array.

        Args:
            only_feature: the only feature to show the structure of if not None.
                Shows all features if None.
        """
        self.expand_to_2d().show_features(only_feature)

    def set_to_use_rcwa(self):
        """Skips the metamodel and directly simulate the atom array using RCWA.

        Note that this method will change the atom array permanently.
        This method is useful for verifying the performance of the metamodel.
        """
        if not self.use_mmodel:
            print("The atom array is already using RCWA simulation directly.")
            return
        metamodel = self.mmodel
        protocell = metamodel.protocell
        self.proto_unit_cell = protocell
        self.use_mmodel = False
        self.cached_fields = None
        self.sim_config = metamodel.sim_config


@dataclasses.dataclass
class Surface:
    """Defines an optical surface.

    Args:
        diameter: the diameter of the surface in meters.
        refractive_index: the refractive index of the surface.
        thickness: the thickness of the surface in meters.
    """

    diameter: float
    refractive_index: float
    thickness: float

    def optimizer_hook(self):
        """Hook for the optimizer to modify the surface."""
        pass

    def get_penalty(self):
        """Returns the penalty of the surface. This is used for the optimizer."""
        # Dummy value, should be overriden by the child class
        return 0.0


@dataclasses.dataclass
class Aperture(Surface):
    """Defines an aperture.

    Args:
        diameter: the diameter of the aperture in meters.
        refractive_index: the refractive index of the aperture.
        thickness: the thickness of the aperture in meters.
        periodicity: the period of the pixels in meters.
        enable_propagator_cache: whether to enable the propagator cache.
            If enabled, the propagator will be cached for the `Incidence`,
            The propagation would be a lot faster however at the cost of
            memory usage. Note that this is only useful when the aperture
            is not moving.
        store_end_field: whether to store the end field of the aperture.

    """

    periodicity: float
    enable_propagator_cache: bool = False
    store_end_field: bool = False

    def __post_init__(self):
        """Intializes the metasurface structure."""
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)

        # Initialize the propagator cache
        self.propagator_cache = (None, None)

        # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
        def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
            """Creates a circular mask."""
            if center is None:  # use the middle of the image
                center = (int(w / 2), int(h / 2))
            if (
                radius is None
            ):  # use the smallest distance between the center and image walls
                radius = min(
                    center[0], center[1], w - center[0], h - center[1]
                )
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt(
                (X - center[0]) ** 2 + (Y - center[1]) ** 2
            )
            mask = dist_from_center <= radius
            return mask

        width = self.n_pixels_radial * 2
        mask = create_circular_mask(width, width)
        self.mask = tf.cast(mask, tf.complex64)
        self.mask = tf.expand_dims(self.mask, axis=0)

    def optimizer_hook(self):
        pass

    def get_modulation_2d(self, incidence: Incidence) -> propagation.Field2D:
        """Computes the field modulation of the metasurface.

        Args:
            incidence: the `Incidence` of the light.

        Returns:
            tf.Tensor: the modulation field with shape (batch_size, n_pixels, n_pixels)
        """

        # Repeat the tensor to match the batch size
        batch_size = (
            len(incidence.wavelength)
            * len(incidence.theta)
            * len(incidence.phi)
        )
        return tf.repeat(self.mask, batch_size, axis=0)

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field at the end of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            incident_field: the incident field.
            previous_refractive_index: the refractive index of the previous
            lateral_shift: the lateral shift of the sampling window on the detector
                in meters. If None, the shift is set so that the Chief Ray is at the
                center of the detector. If a tuple of two floats, the shift is set
                according to the first element (x shift) and the second element (y
                shift) of the input tuple, in meters.
            last_surface: whether this is the last surface in the optical system.
            use_padding: whether to use padding for the field.
        """
        mod_tensor = self.get_modulation_2d(incidence)
        mod_field = propagation.Field2D(
            tensor=mod_tensor,
            period=self.periodicity,
            n_pixels=mod_tensor.shape[-1],
            wavelength=incidence.wavelength,
            theta=incidence.theta,
            phi=incidence.phi,
            upsampling=1,
            use_antialiasing=True,
            use_padding=use_padding,
        )
        mod_field = mod_field.modulated_by(incident_field)

        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = mod_field
            return mod_field

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=mod_field,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=mod_field,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        end_field = propagation.propagate(mod_field, propagator)
        if self.store_end_field:
            self.end_field = end_field

        return end_field


@dataclasses.dataclass
class AmplitudeMask(Surface):
    """Defines an amplitude modulation mask.

    Args:
        diameter: the diameter of the lens in meters.
        refractive_index: the refractive index of the lens.
        thickness: the thickness of the lens in meters.
        periodicity: the period of the pixels in meters.
        threshold_param: the threshold parameter for the amplitude modulation.
            This param multiples the amplitude coefficient before the sigmoid
            function. The larger the value, the more "black and white" the
            thresholding is.
        threshold_param_increment: the increment of the threshold parameter
            when the optimizer_hook() is called.
        enable_propagator_cache: whether to enable the propagator cache.
            If enabled, the propagator will be cached for the `Incidence`,
            The propagation would be a lot faster however at the cost of
            memory usage. Note that this is not recommended if the
            `Incidence` is not fixed.
        set_mask_variable: whether to make the mask variable.
        store_end_field: whether to store the end field.
    """

    periodicity: float
    threshold_param: float
    use_circular_expansions: bool = True
    enable_propagator_cache: bool = False
    set_mask_variable: bool = False
    threshold_param_increment: float = 0.0
    store_end_field: bool = False

    def __post_init__(self):
        """Intializes the metasurface structure."""
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)
        if self.use_circular_expansions:
            self.coeff_1d = initialize_1d_mask_array(
                self.n_pixels_radial,
                self.set_mask_variable,
            )
        else:
            self.coeff_2d = initialize_2d_mask_array(
                self.n_pixels_radial,
                self.set_mask_variable,
            )

        # Initialize the propagator cache
        self.propagator_cache = (None, None)

        # Add the variables to the list of variables
        self.variables = []
        if self.set_mask_variable:
            self.variables.append(self.coeff_1d)
        # TODO: add the 2d coeff to the variables

    def optimizer_hook(self):
        """Hook for the optimizer to modify the surface."""
        # Update the threshold parameter
        self.threshold_param += self.threshold_param_increment

    def get_modulation_2d(self, incidence: Incidence) -> propagation.Field2D:
        """Computes the field modulation of the metasurface.

        Args:
            incidence: the `Incidence` of the light.

        Returns:
            tf.Tensor: the modulation field with shape (batch_size, n_pixels, n_pixels)
        """
        new_tensor = expansion.expand_to_2d(
            self.coeff_1d[tf.newaxis, :], "basis_data"
        )
        # Apply the thresholding
        new_tensor = self.threshold_param * new_tensor
        new_tensor = tf.math.sigmoid(new_tensor)

        # Repeat the tensor to match the batch size
        batch_size = (
            len(incidence.wavelength)
            * len(incidence.theta)
            * len(incidence.phi)
        )
        new_tensor = tf.repeat(new_tensor, batch_size, axis=0)
        return new_tensor

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field at the end of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            incident_field: the incident field.
            previous_refractive_index: the refractive index of the previous
            lateral_shift: the lateral shift of the sampling window on the detector
                in meters. If None, the shift is set so that the Chief Ray is at the
                center of the detector. If a tuple of two floats, the shift is set
                according to the first element (x shift) and the second element (y
                shift) of the input tuple, in meters.
            last_surface: whether this is the last surface in the optical system.
            use_padding: whether to use padding for the field.
        """
        mod_tensor = self.get_modulation_2d(incidence)
        mod_field = propagation.Field2D(
            tensor=mod_tensor,
            period=self.periodicity,
            n_pixels=mod_tensor.shape[-1],
            wavelength=incidence.wavelength,
            theta=incidence.theta,
            phi=incidence.phi,
            upsampling=1,
            use_antialiasing=True,
            use_padding=use_padding,
        )
        mod_field = mod_field.modulated_by(incident_field)

        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = mod_field
            return mod_field

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=mod_field,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=mod_field,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        end_field = propagation.propagate(mod_field, propagator)
        if self.store_end_field:
            self.end_field = end_field

        return end_field


@dataclasses.dataclass
class SphericalLens(Surface):
    perioidicity: float
    radius_or_curvature: float
    """Defines a spherical lens.

    Args:
        periodicity: the period of the lens in meters.
        radius_or_curvature: the radius of curvature of the lens in meters.

    Returns:
        _type_: _description_
    """

    def __post_init__(self):
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)
        raise NotImplementedError

    def get_modulation_2d(
        incidence: Incidence,
    ):
        pass

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ):
        pass


@dataclasses.dataclass
class RefractiveEvenAsphere(Surface):
    """Defines an even asphere surface comparable to Zemax.

    The even asphere surfaces use polynomial terms to express the sag surface.
            z = Σ{i=1; N} (A_i * ρ**(2 * i))
        N is the maximum number of terms, we don't have restrictions here, but
        Zemax limits the number of terms to 8. The extended asphere supports
        up to 480 terms. A_i is the coefficient of the ith term, and ρ is the
        normalized radial coordinate of the aperture.

    Args:
        diameter: the diameter of the surface in meters.
        refractive_index: the refractive index of the surface.
        thickness: the thickness of the surface in meters.
        periodicity: the period of the surface in meters.
        unit: the unit used in Zemax. Can be "m" or "mm". Defaults to "m".
        set_coeff_variable: whether to set the coefficients as variables.
        enable_propagator_cache: whether to enable the propagator cache.
        store_end_field: whether to store the end field of the surface.
        thickness_penalty_coeff: the coefficient of the thickness penalty term.
            Multiplied to the maximum thickness of the sag as the penalty.
    """

    periodicity: float
    init_coeff: List[float] = None
    set_coeff_variable: bool = True
    enable_propagator_cache: bool = False
    store_end_field: bool = False
    thickness_penalty_coeff: float = 1e-3

    def __post_init__(self):
        """Initialization"""
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)
        self.coeff = tf.cast(self.init_coeff, dtype=tf.float32)
        if self.set_coeff_variable:
            self.coeff = tf.Variable(
                initial_value=self.coeff,
                trainable=True,
                dtype=tf.float32,
                name="even_asphere_coeff",
            )

        # Initialize the propagator cache
        self.propagator_cache = (None, None)

        # Add the variables to the list of variables
        self.variables = []
        if self.set_coeff_variable:
            self.variables.append(self.coeff)

    def get_sag(self):
        """Returns the sag surface"""
        scale = 1e3  # Zemax scales the coefficients by 1e3 for some reason.
        radius = self.diameter / 2.0
        rho = tf.linspace(0.0, radius, self.n_pixels_radial)
        rho = tf.cast(rho, dtype=tf.float32)
        sag = tf.zeros(self.n_pixels_radial, dtype=tf.float32)
        for i, A_i in enumerate(self.coeff):
            sag += A_i * scale * tf.pow(rho, 2 * (i + 1))
        return sag

    def get_penalty(self):
        """Returns the penalty of the surface. This is used for the optimizer."""
        max_thickness = tf.abs(tf.reduce_max(self.get_sag()))
        return (
            tf.math.log(max_thickness + 1e-12) * self.thickness_penalty_coeff
        )

    def show_sag(self):
        sag = self.get_sag()
        # Get the other half
        sag = tf.concat([tf.reverse(sag, axis=[0]), sag], axis=0)
        sag = tf.cast(sag, dtype=tf.float64)
        diameter = self.periodicity * sag.shape[0]
        radius = diameter / 2.0
        dist = np.linspace(-radius, radius, sag.shape[0])
        f = plt.figure(figsize=(8, 5), dpi=100)
        ax = plt.axes([0, 0, 1.0, 1.0])
        im = ax.plot(dist, sag)
        formatter0 = EngFormatter(unit="m")
        ax.xaxis.set_major_formatter(formatter0)
        ax.yaxis.set_major_formatter(formatter0)
        plt.locator_params(axis="y", nbins=8)
        plt.locator_params(axis="x", nbins=3)
        ax.set_xlabel("Distance from the center")
        ax.set_ylabel("Sag")
        title = "Surface Sag Cross-section"
        ax.set_title(title)
        plt.show()

    def get_modulation_2d(
        self,
        incidence: Incidence,
        previous_refractive_index: float,
        use_padding: bool = True,
    ) -> propagation.Field2D:
        """Computes the field modulation of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            previous_refractive_index: the refractive index of the previous
            use_padding: whether to use padding.

        Returns:
            propagation.Field2D: the field modulation of the metasurface.
        """
        sag = self.get_sag()
        batch_size = (
            len(incidence.theta)
            * len(incidence.phi)
            * len(incidence.wavelength)
        )
        sag = tf.repeat(sag[tf.newaxis, :], batch_size, axis=0)
        wavelength = tf.convert_to_tensor(
            np.repeat(
                incidence.wavelength,
                np.size(incidence.theta) * np.size(incidence.phi),
            ),
            dtype=tf.float32,
        )
        wavelength = wavelength[:, tf.newaxis]
        delta_n = self.refractive_index - previous_refractive_index
        phi = sag * delta_n * 2 * np.pi / wavelength
        phi = tf.cast(phi, dtype=tf.complex64)
        field = tf.exp(-1j * phi)

        field_1d = propagation.Field1D(
            tensor=field,
            n_pixels=self.n_pixels_radial * 2,
            wavelength=incidence.wavelength,
            theta=incidence.theta,
            phi=incidence.phi,
            period=self.periodicity,
            upsampling=1,
            use_padding=use_padding,
            use_antialiasing=True,
        )

        return field_1d.expand_to_2d()

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field at the end of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            incident_field: the incident field.
            previous_refractive_index: the refractive index of the previous surface.
            lateral_shift: the lateral shift of the sampling window on the detector
                in meters. If None, the shift is set so that the Chief Ray is at the
                center of the detector. If a tuple of two floats, the shift is set
                according to the first element (x shift) and the second element (y
                shift) of the input tuple, in meters.
            use_padding: whether to use padding to avoid aliasing.
        """
        field_2d = self.get_modulation_2d(
            incidence,
            previous_refractive_index,
            use_padding,
        )
        field_2d = field_2d.modulated_by(incident_field)
        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = field_2d
            return field_2d

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=field_2d,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=field_2d,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        field_2d = propagation.propagate(field_2d, propagator)
        if self.store_end_field:
            self.end_field = field_2d

        return field_2d


@dataclasses.dataclass
class Binary2(Surface):
    """Defines a binary 2 surface comparable to the namesake surface in Zemax.

    The binary 2 surface use polynomial terms to express the phase delay of the
        incident field. The phase delay Φ is given by:
            Φ = M * Σ{i=1; N} (A_i * ρ**(2 * i))
        Where M is the diffraction order, N is the maximum number of terms, A_i is
        the coefficient of the ith term, and ρ is the normalized radial
        coordinate of the aperture.

    Args:
        diameter: the diameter of the surface in meters.
        refractive_index: the refractive index of the surface.
        thickness: the thickness of the surface in meters.
        periodicity: the period of the surface in meters.
        diffraction_order: the diffraction order of the surface.
        store_end_field: whether to store the end field of the surface.
        previous_refractive_index: the refractive index of the previous surface.
    """

    periodicity: float
    init_coeff: List[float] = None
    set_coeff_variable: bool = True
    enable_propagator_cache: bool = False
    diffraction_order: int = 1
    store_end_field: bool = False
    previous_refractive_index: float = 1.0

    def __post_init__(self):
        """Initialization"""
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)
        self.coeff = tf.cast(self.init_coeff, dtype=tf.float32)
        if self.set_coeff_variable:
            self.coeff = tf.Variable(
                initial_value=self.coeff,
                trainable=True,
                dtype=tf.float32,
                name="binary2coeff",
            )

        # Initialize the propagator cache
        self.propagator_cache = (None, None)

        # Add the variables to the list of variables
        self.variables = []
        if self.set_coeff_variable:
            self.variables.append(self.coeff)

    def get_modulation_2d(
        self, incidence: Incidence, use_padding: bool = True
    ) -> propagation.Field2D:
        """Computes the field modulation of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            use_padding: whether to use padding.
        """
        rho = tf.linspace(0, 1, self.n_pixels_radial)
        rho = tf.cast(rho, dtype=tf.float32)
        phi = tf.zeros(self.n_pixels_radial, dtype=tf.float32)
        for i, A_i in enumerate(self.coeff):
            phi += A_i * tf.pow(rho, 2 * (i + 1))
        phi = self.diffraction_order * phi
        batch_size = (
            len(incidence.theta)
            * len(incidence.phi)
            * len(incidence.wavelength)
        )
        phi = tf.repeat(phi[tf.newaxis, :], batch_size, axis=0)
        phi = tf.cast(phi, dtype=tf.complex64)
        field = tf.exp(1j * phi)

        field_1d = propagation.Field1D(
            tensor=field,
            n_pixels=self.n_pixels_radial * 2,
            wavelength=incidence.wavelength,
            theta=incidence.theta,
            phi=incidence.phi,
            period=self.periodicity,
            upsampling=1,
            use_padding=use_padding,
            use_antialiasing=True,
        )

        return field_1d.expand_to_2d()

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field at the end of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            incident_field: the incident field.
            previous_refractive_index: the refractive index of the previous
                surface.
            lateral_shift: the lateral shift of the sampling window on the detector
                in meters. If None, the shift is set so that the Chief Ray is at the
                center of the detector. If a tuple of two floats, the shift is set
                according to the first element (x shift) and the second element (y
                shift) of the input tuple, in meters.
            use_padding: whether to use padding.
        """
        field_2d = self.get_modulation_2d(incidence, use_padding=use_padding)
        field_2d = field_2d.modulated_by(incident_field)
        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = field_2d
            return field_2d

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=field_2d,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=field_2d,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        field_2d = propagation.propagate(field_2d, propagator)
        if self.store_end_field:
            self.end_field = field_2d

        return field_2d


@dataclasses.dataclass
class Metasurface(Surface):
    """Defines a flat lens.

    Args:
        diameter: the diameter of the lens in meters.
        refractive_index: the refractive index of the lens.
        thickness: the thickness of the lens in meters.
        metamodel: the metamodel used to map the structure to the field
            modulation for each meta-atom.
        enable_propagator_cache: whether to enable the propagator cache.
            If enabled, the propagator will be cached for the `Incidence`,
            The propagation would be a lot faster however at the cost of
            memory usage. Note that this is not recommended if the
            `Incidence` is not fixed.
        set_structures_variable: whether to make the structures variable.
        store_end_field: whether to store the end field.
        xy_harmonics: the number of harmonics in the x and y directions for the
            field modulation.
        unit_cell_spatial_res: the spatial resolution of the unit cell.
        minibatch_size: the minibatch size for the rcwa simulation.
    """

    metamodel: modeling.Metamodel = None
    proto_unit_cell: rcwa.ProtoUnitCell = None
    use_circular_expansions: bool = True
    enable_propagator_cache: bool = False
    set_structures_variable: bool = False
    store_end_field: bool = False
    xy_harmonics: Tuple[int, int] = (3, 3)
    unit_cell_spatial_res: int = 128
    minibatch_size: int = 100

    def __post_init__(self):
        """Intializes the metasurface structure."""
        has_metamodel = self.metamodel is not None
        has_proto_unit_cell = self.proto_unit_cell is not None
        if not has_metamodel and not has_proto_unit_cell:
            raise ValueError(
                "Neither `metamodel` nor `proto_unit_cell` is specified."
                "Either `metamodel` or `proto_unit_cell` must be specified."
            )
        if has_metamodel and has_proto_unit_cell:
            raise ValueError(
                "Both `metamodel` and `proto_unit_cell` are specified."
                "Only ONE of `metamodel` or `proto_unit_cell` must be specified."
            )
        self.sim_config = rcwa.SimConfig(
            xy_harmonics=self.xy_harmonics,
            resolution=self.unit_cell_spatial_res,
            minibatch_size=self.minibatch_size,
            return_tensor=True,
            return_zeroth_order=True,
            use_transmission=True,
            include_z_comp=False,
        )
        self.use_metamodel = has_metamodel
        if self.use_metamodel:
            self.periodicity = (
                self.metamodel.protocell.proto_unit_cell.periodicity[0]
            )
        else:
            self.periodicity = (
                self.proto_unit_cell.proto_unit_cell.periodicity[0]
            )
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)

        if self.use_metamodel:
            args = (
                self.n_pixels_radial,
                self.periodicity,
                self.metamodel,
                self.set_structures_variable,
            )
            if self.use_circular_expansions:
                self.atom_1d = initialize_1d_atom_array_metamodel(*args)
            else:
                self.atom_2d = initialize_2d_atom_array_metamodel(*args)
        else:
            args = (
                self.n_pixels_radial,
                self.proto_unit_cell,
                self.set_structures_variable,
            )
            if self.use_circular_expansions:
                self.atom_1d = initialize_1d_atom_array_proto_unit_cell(*args)
                self.atom_1d.sim_config = self.sim_config
            else:
                self.atom_2d = initialize_2d_atom_array_proto_unit_cell(*args)
                self.atom_2d.sim_config = self.sim_config

        # Initialize the propagator cache
        self.propagator_cache = (None, None)

        # Add the variables to the list of variables
        self.variables = []
        if self.set_structures_variable:
            if self.use_circular_expansions:
                self.variables.append(self.atom_1d.tensor)
            else:
                self.variables.append(self.atom_2d.tensor)

    def get_atom_positions(self) -> np.ndarray:
        """Gets the positions of the atoms."""
        x_pos = y_pos = np.linspace(
            -self.diameter / 2, self.diameter / 2, 2 * self.n_pixels_radial
        )
        xx_pos, yy_pos = np.meshgrid(x_pos, y_pos)
        return np.stack([xx_pos, yy_pos], axis=-1)

    def get_modulation_2d(
        self,
        incidence: Incidence,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field modulation of the metasurface."""
        if self.use_circular_expansions:
            if self.atom_1d.cached_fields is None:
                field_1d_x, field_1d_y = structure_to_field_1d(
                    self.atom_1d,
                    incidence,
                    use_padding=use_padding,
                )
                self.atom_1d.cached_fields = field_1d_x, field_1d_y

            field_1d_x, field_1d_y = self.atom_1d.cached_fields
            if use_x_pol:
                output_field = field_1d_x.expand_to_2d()
            else:
                output_field = field_1d_y.expand_to_2d()

        else:
            if self.atom_2d.cached_fields is None:
                self.atom_2d.cached_fields = structure_to_field_2d(
                    self.atom_2d,
                    incidence,
                    use_padding=use_padding,
                )
            if use_x_pol:
                output_field = self.atom_2d.cached_fields[0]
            else:
                output_field = self.atom_2d.cached_fields[1]

        return output_field

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: Union[None, Tuple[float, float]] = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        """Computes the field at the end of the metasurface.

        Args:
            incidence: the `Incidence` of the light.
            incident_field: the incident field.
            previous_refractive_index: the refractive index of the previous
                medium.
            lateral_shift: the lateral shift of the sampling window on the detector
                in meters. If None, the shift is set so that the Chief Ray is at the
                center of the detector. If a tuple of two floats, the shift is set
                according to the first element (x shift) and the second element (y
                shift) of the input tuple, in meters.
            use_padding: whether to use padding.
        """
        field_2d = self.get_modulation_2d(
            incidence, use_padding=use_padding, use_x_pol=use_x_pol
        )
        field_2d = field_2d.modulated_by(incident_field)
        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = field_2d
            return field_2d

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=field_2d,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=field_2d,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        field_2d = propagation.propagate(field_2d, propagator)
        if self.store_end_field:
            self.end_field = field_2d

        return field_2d

    def get_atom_arry(self, incidence: Incidence) -> List[rcwa.UnitCell]:
        """Gets the atom array for the given incidence."""
        return self.atom_1d.get_atom_array(incidence)

    def get_feature_map(self):
        """Gets the feature map of the metasurface."""
        if self.use_circular_expansions:
            return self.atom_1d.get_feature_map()
        else:
            return self.atom_2d.get_feature_map()

    def set_feature_map(
        self, feature_str: str, new_value: Union[np.ndarray, tf.Tensor]
    ):
        """Sets the feature map of the metasurface.

        Args:
            feature_str: the feature to set.
            new_value: the new value of the feature.
        """
        if self.use_circular_expansions:
            self.atom_1d.set_feature_map(feature_str, new_value)
        else:
            self.atom_2d.set_feature_map(feature_str, new_value)

    def show_feature_map(self, only_feature: Union[str, None] = None):
        """Shows the feature map of the metasurface.

        Args:
            only_feature: if not None, only shows the feature map of the given
                feature. Otherwise, shows the featue map of all features.
        """
        if self.use_circular_expansions:
            self.atom_1d.show_feature_map(only_feature=only_feature)
        else:
            self.atom_2d.show_feature_map(only_feature=only_feature)

    def optimizer_hook(self):
        self.clear_cache()

    def set_to_use_rcwa(self):
        """Set to use RCWA for the metasurface, permanently."""
        if self.use_circular_expansions:
            self.atom_1d.set_to_use_rcwa()
        else:
            self.atom_2d.set_to_use_rcwa()

    def clear_cache(self):
        """Clears the saved fields."""
        if self.use_circular_expansions:
            self.atom_1d.cached_fields = None
        else:
            self.atom_2d.cached_fields = None


@enum.unique
class FigureOfMerit(enum.Enum):
    """Defines the types of figure of merit functions.

    Attributes:
        STREHL_RATIO: the Strehl ratio.
        LOG_STREHL_RATIO: the log of the Strehl ratio.
    """

    STREHL_RATIO = 1
    LOG_STREHL_RATIO = 2
    MAX_INTENSITY = 3
    LOG_MAX_INTENSITY = 4
    CENTER_INTENSITY = 5
    LOG_CENTER_INTENSITY = 6


@dataclasses.dataclass
class CustomFigureOfMerit:
    f"""A data class that represents a custom figure of merit function.

    This class encapsulates a mathematical expression that represents a figure of merit
    function. It ensures the validity of the expression based on predefined constraints,
    and raises exceptions if the given expression violates them.

    Allowed operators: +, -, *, /, (, )
    Allowed predefined variable names:
        psf: the point spread function. Shape: (batch, n_pixels, n_pixels)
        strehl_ratio: the Strehl ratio. Shape: (batch,)
        max_intensity: the maximum intensity. Shape: (batch,)
        center_intensity: the center intensity. Shape: (batch,)
        ideal_mtf: the ideal modulation transfer function. Shape: (batch, n_pixels, n_pixels)
    Allowed functions: {utils.TF_FUNCTIONS}

    Attributes:
        expression (str): The mathematical expression that represents the figure of merit.

    Methods:
        get_validation_errors: Checks the validity of the expression based on predefined
        constraints and returns detailed error messages for any violations.

        is_valid_expression: Validates the mathematical structure and content of the
        expression against allowed patterns.

    Example:
        >>> fom = CustomFigureOfMerit("psf + strehl_ratio")

    Args:
        expression: the expression of the figure of merit function.
        data: user can provide extra data to be used in the expression.
            For instance, user can define an expression such as
            "reduce_sum((psf - target_image)**2)" where
            CustomFigureOfMerit.data["target_image"] is a tensor of shape
            (batch, n_pixels, n_pixels).
    """

    expression: str
    data: Dict[str, tf.Tensor] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Initialization"""
        if self.expression is None:
            raise ValueError("The expression cannot be None.")
        validation_errors = self.get_validation_errors()
        if validation_errors:
            raise ValueError(
                f"Expression validation failed:\n{validation_errors}"
            )

    def get_validation_errors(self):
        allowed_keywords = [
            "psf",
            "strehl_ratio",
            "max_intensity",
            "center_intensity",
            "ideal_mtf",
            "dist",
        ]
        validation_errors = []

        # Include keys from the user data in the allowed keywords
        allowed_keywords.extend(self.data.keys())

        for keyword in allowed_keywords:
            if keyword in self.expression and not re.search(
                rf"\b{keyword}\b", self.expression
            ):
                validation_errors.append(
                    f"'{keyword}' should not be part of another word or surrounded by non-allowed characters."
                )

        if not self.is_valid_expression(self.expression):
            validation_errors.append(
                "Ensure you are only using allowed operators: +, -, *, /, (, ), :.\n"
                f"And the following functions: {utils.TF_FUNCTIONS + 'log'}.\n"
                "Ensure you are using the allowed variables: psf, strehl_ratio, max_intensity, center_intensity, ideal_mtf."
            )

        return "\n".join(validation_errors)

    def is_valid_expression(self, user_expression):
        # List of allowed characters and patterns
        allowed_patterns = [
            r"\bpsf\b",
            r"\bstrehl_ratio\b",
            r"\bmax_intensity\b",
            r"\bcenter_intensity\b",
            r"\bideal_mtf\b",
            r"\blog\b",
            r"\+",
            r"\-",
            r"\/",
            r"\(",
            r"\)",
            r"\:",
            r"\...",
            r"\d+(\.\d+)?",  # basic patterns
        ]

        # Add the keys from the user data to the allowed patterns
        allowed_patterns.extend(self.data.keys())

        tf_functions = utils.TF_FUNCTIONS
        allowed_patterns.extend(tf_functions)

        # Construct the regex pattern
        pattern = "|".join(allowed_patterns)

        # Split the user expression into elements
        elements = re.split(r"\s|(?<=[\(\)\+\-\*/])", user_expression)

        # Check each element against the patterns
        for element in elements:
            if element and not re.match(pattern, element):
                return False

        return True


@dataclasses.dataclass
class LensAssembly:
    """Defines a lens assembly.

    Args:
        surfaces: a list of surfaces in the lens assembly.
        focal_length: the focal length of the lens assembly in meters.
        aperture_stop_index: the index of the aperture stop in the lens assembly.
        figure_of_merit: the figure of merit of the lens assembly.
            Options can be found in the `FigureOfMerit` enum.
        use_antialiasing: whether to use antialiasing for propagations.
        use_padding: whether to use padding for propagations. If True, the
            sampling window is padded to avoid aliasing at the cost of
            ~4x memory usage.
        use_x_pol: whether the lens assembly is sensitive to the x polarization.
            if True, the x polarization is used. Otherwise, the y polarization
            is used.
    """

    surfaces: List[Surface]
    incidence: Incidence
    aperture_stop_index: int = -1
    figure_of_merit: Union[FigureOfMerit, CustomFigureOfMerit, None] = None
    use_antialiasing: bool = True
    use_padding: bool = True
    use_x_pol: bool = True

    def __post_init__(self):
        # Not a parameter at the moment as it is not very useful and this
        # feature is not thoroughly tested yet.
        self.upsampling = 1
        # Focal length of the lens assembly.
        focal_length = 0
        for a_surface in self.surfaces[self.aperture_stop_index :]:
            focal_length += a_surface.thickness

        # Calculate the ideal volume of MTF.
        ref_surface = self.surfaces[self.aperture_stop_index]
        n_pixels = ref_surface.n_pixels_radial * 2

        # Define the field properties.
        self.field_properties = propagation.FieldProperties(
            n_pixels=n_pixels,
            wavelength=self.incidence.wavelength,
            theta=self.incidence.theta,
            phi=self.incidence.phi,
            period=ref_surface.periodicity,
            upsampling=self.upsampling,
            use_antialiasing=self.use_antialiasing,
            use_padding=self.use_padding,
        )

        if self.figure_of_merit is not None:
            self.ideal_mtf = metrics.get_ideal_mtf_volume(
                field_props=self.field_properties,
                focal_length=focal_length,
            )

    def compute_field_on_sensor(self):
        """Computes the Strehl ratio of the lens assembly."""
        current_field = propagation.get_incident_field_2d(
            self.field_properties
        )
        for idx, surface in enumerate(self.surfaces):
            if idx == len(self.surfaces) - 1:
                lateral_shift = None  # for the last surface
            else:
                lateral_shift = (0, 0)  # for intermediate surfaces

            if idx == 0:
                previous_refractive_index = 1.0
            else:
                previous_refractive_index = self.surfaces[
                    idx - 1
                ].refractive_index

            # Cascading the fields
            current_field = surface.get_end_field(
                incidence=self.incidence,
                incident_field=current_field,
                previous_refractive_index=previous_refractive_index,
                lateral_shift=lateral_shift,
                use_padding=self.use_padding,
                use_x_pol=self.use_x_pol,
            )
        return current_field

    def show_psf(
        self, use_wavelength_average: bool = False, crop_factor: float = 1.0
    ) -> None:
        """Displays the point spread function of the lens assembly.

        Args:
            use_wavelength_averaging: whether to use wavelength averaging.
            crop_factor: the crop factor of the image.
        """
        if use_wavelength_average:
            self.compute_field_on_sensor().wavelength_average().show_intensity(
                crop_factor=crop_factor
            )
        else:
            self.compute_field_on_sensor().show_intensity(
                crop_factor=crop_factor
            )
        self.clear_cache()

    def show_color_psf(
        self,
        crop_factor: float = 1.0,
    ) -> None:
        self.compute_field_on_sensor().show_color_intensity(
            crop_factor=crop_factor
        )
        self.clear_cache()

    def wavelength_average_psf(self):
        """Displays the wavelength averaged point spread function of the lens assembly."""

    def compute_strehl_ratio(self):
        """Computes the Strehl ratio of the lens assembly."""
        field = self.compute_field_on_sensor()
        return metrics.get_mtf_volume(field) / self.ideal_mtf[:, tf.newaxis]

    def compute_max_intensity(self):
        """Computes the maximum intensity of the lens assembly."""
        field = self.compute_field_on_sensor()
        return metrics.get_max_intensity(field)

    def compute_center_intensity(self):
        """Computes the center intensity of the lens assembly."""
        field = self.compute_field_on_sensor()
        return metrics.get_center_intensity(field)

    def get_variables(self):
        """Returns the trainable variables."""
        variables = []
        for surface in self.surfaces:
            variables += surface.variables
        return variables

    def compute_FOM(self) -> tf.Tensor:
        """Computes the figure of merit of the lens assembly.

        Args:
            tf.Tensor: The figure of merit.
        """
        if self.figure_of_merit is None:
            raise ValueError("No figure of merit defined.")
        elif isinstance(self.figure_of_merit, CustomFigureOfMerit):
            return self.compute_custom_FOM(self.figure_of_merit)
        elif self.figure_of_merit not in FigureOfMerit:
            raise ValueError(
                f"Invalid figure of merit {self.figure_of_merit}."
            )

        elif self.figure_of_merit == FigureOfMerit.STREHL_RATIO:
            return tf.reduce_mean(self.compute_strehl_ratio())
        elif self.figure_of_merit == FigureOfMerit.LOG_STREHL_RATIO:
            return tf.reduce_mean(tf.math.log(self.compute_strehl_ratio()))
        elif self.figure_of_merit == FigureOfMerit.MAX_INTENSITY:
            return tf.reduce_mean(self.compute_max_intensity())
        elif self.figure_of_merit == FigureOfMerit.LOG_MAX_INTENSITY:
            return tf.reduce_mean(tf.math.log(self.compute_max_intensity()))
        elif self.figure_of_merit == FigureOfMerit.CENTER_INTENSITY:
            return tf.reduce_mean(self.compute_center_intensity())
        elif self.figure_of_merit == FigureOfMerit.LOG_CENTER_INTENSITY:
            return tf.reduce_mean(tf.math.log(self.compute_center_intensity()))
        else:
            raise ValueError(
                "Invalid figure of merit. This should never happen."
            )

    def compute_custom_FOM(self, custom_FOM: CustomFigureOfMerit) -> tf.Tensor:
        # List of tensorflow functions (you can extend this list as needed)
        tf_functions = utils.TF_FUNCTIONS

        user_expression = custom_FOM.expression

        # Replace arithmetic operations
        replacements = {
            "\*": " * ",
            "\/": " / ",
            "\+": " + ",
            "\-": " - ",
        }
        for pattern, replacement in replacements.items():
            user_expression = re.sub(pattern, replacement, user_expression)

        # Compute variables if needed
        if "psf" in user_expression:
            psf = self.compute_field_on_sensor()
        if "strehl_ratio" in user_expression:
            if "psf" not in locals():
                psf = self.compute_field_on_sensor()
            strehl_ratio = metrics.get_mtf_volume(psf) / self.ideal_mtf
        if "max_intensity" in user_expression:
            if "psf" not in locals():
                psf = self.compute_field_on_sensor()
            max_intensity = metrics.get_max_intensity(psf)
        if "center_intensity" in user_expression:
            if "psf" not in locals():
                psf = self.compute_field_on_sensor()
            center_intensity = metrics.get_center_intensity(psf)
        if "psf" in locals():
            psf_tensor = tf.math.abs(psf.tensor) ** 2

        replacements = {
            "ideal_mtf": "self.ideal_mtf",
            "psf": "psf_tensor",
            "log": "tf.math.log",
        }

        for old, new in replacements.items():
            user_expression = user_expression.replace(old, new)

        # Replace functions with TensorFlow functions
        for func in tf_functions:
            user_expression = user_expression.replace(func, f"tf.{func}")

        # Add the user data to the local variables
        if custom_FOM.data:
            for key, value in custom_FOM.data.items():
                user_expression = user_expression.replace(
                    key, f"custom_FOM.data['{key}']"
                )

        # Evaluate the TensorFlow expression
        return eval(user_expression)

    def compute_penalty(self) -> tf.Tensor:
        """Computes the penalty of the lens assembly.

        Args:
            tf.Tensor: The penalty.
        """
        penalty = 0
        for surface in self.surfaces:
            penalty += surface.get_penalty()
        return penalty

    def copy(self) -> "LensAssembly":
        """Returns a copy of the lens assembly.

        Returns:
            LensAssembly: The copy of the lens assembly.
        """
        # copy.deepcopy doesn't work for serializing tf models.
        return copy_lens_assembly(self)

    def save(
        self,
        name: str,
        save_dir: str = "./saved_lens_assemblies",
        overwrite: bool = False,
    ):
        """Saves the lens assembly to disk.

        Args:
            name: the name of the lens assembly.
            save_dir: the directory to save the lens assembly to.
        """
        save_lens_assembly(self, name, save_dir, overwrite)

    def optimizer_hook(self):
        """Hook for the optimizer iteration."""
        for surface in self.surfaces:
            surface.optimizer_hook()

    def set_to_use_rcwa(self):
        """Use RCWA simulation for all the metasurfaces, permanently.

        Note that this function will permanently change the metasurfaces to use
        RCWA simulation. It's wise to save the lens assembly before calling this
        function. Or make a copy of the lens assembly before calling this.
        """
        for surface in self.surfaces:
            if isinstance(surface, Metasurface):
                surface.set_to_use_rcwa()

    def clear_cache(self):
        """Clears saved fields."""
        for surface in self.surfaces:
            if type(surface) is Metasurface:
                surface.clear_cache()


@dataclasses.dataclass
class IntensityTarget:
    intensity: tf.Tensor
    crop_factor: float = 1.0

    def __post_init__(self):
        self.intensity = tf.cast(self.intensity, dtype=tf.float32)

        # # Pad to make it a square
        # shape = tf.shape(self.intensity)

        # # Calculate padding
        # dim_diff = tf.abs(shape[0] - shape[1]) // 2
        # lower_pad = dim_diff
        # upper_pad = (
        #     dim_diff if tf.shape(self.intensity)[0] % 2 == 0 else dim_diff + 1
        # )

        # rows_pad = (lower_pad, upper_pad) if shape[0] < shape[1] else (0, 0)
        # cols_pad = (lower_pad, upper_pad) if shape[0] > shape[1] else (0, 0)

        # # Pad the tensor
        # self.intensity = tf.pad(
        #     self.intensity, [rows_pad, cols_pad], "CONSTANT"
        # )

        # Normalize the intensity
        self.intensity = self.intensity / tf.reduce_sum(self.intensity)

    def dist(
        self,
        psf: tf.Tensor,
    ) -> tf.Tensor:
        """Computes the loss between the target intensity and the intensity of the field.

        Args:

        """
        return cartesian_distance(self, psf)


def cartesian_distance(
    intensity_target: IntensityTarget,
    psf: tf.Tensor,
):
    """Calculates the distance between the target intensity and the intensity of the field.

    Args:
        intensity_target: the target intensity.
        psf: the point spread function.

    Returns:
        tf.Tensor: the distance between the target intensity and the intensity of
            the field.
    """

    # normalize psf
    psf = psf / tf.reduce_sum(psf)

    # Add two extra dimensions to make it a 4D tensor
    image_4d = tf.expand_dims(
        tf.expand_dims(intensity_target.intensity, axis=0), axis=-1
    )

    # Now you can resize it
    new_height = int(psf.shape[-2] * intensity_target.crop_factor)
    new_width = int(psf.shape[-1] * intensity_target.crop_factor)
    resized_image_4d = tf.image.resize_with_pad(
        image_4d, new_height, new_width
    )
    if intensity_target.crop_factor != 1.0:
        resized_image_4d = tf.image.resize_with_crop_or_pad(
            resized_image_4d, psf.shape[-2], psf.shape[-1]
        )

    # Remove the extra dimensions to get the resized 2D image
    target = tf.squeeze(resized_image_4d, axis=[0, -1])

    # normalize the target intensity
    target = target / tf.reduce_sum(target)

    # calculate the distance
    distance = tf.reduce_sum(tf.math.abs(psf - target) ** 2) ** 0.5
    return distance


def copy_lens_assembly(lens_assembly: LensAssembly) -> LensAssembly:
    """Returns a copy of the lens assembly.

    Args:
        lens_assembly: the lens assembly to copy.

    Returns:
        LensAssembly: the copy of the lens assembly.
    """
    with utils.suppress_stdout_stderr():
        save_lens_assembly(lens_assembly, "temp", "./", overwrite=True)
        return load_lens_assembly("temp", "./")


def save_lens_assembly(
    lens_assembly: LensAssembly,
    name: str,
    save_dir: str = "./saved_lens_assemblies",
    overwrite: bool = False,
) -> None:
    """Saves the lens assembly to disk.

    Args:
        name: the name of the lens assembly.
        save_dir: the directory to save the lens assembly to.
        overwrite: whether to overwrite the lens assembly if it already
            exists.
    """
    save_path = os.path.join(save_dir, name)
    if os.path.exists(save_path):
        if not overwrite:
            raise ValueError(
                f"Lens assembly {name} already exists. Set overwrite=True to "
                "overwrite."
            )
    else:
        os.mkdir(save_path)

    save_path_pkl = os.path.join(save_path, "lens_assembly.pkl")
    with utils.suppress_stdout_stderr():
        new_self = copy.deepcopy(lens_assembly)
    for surface in new_self.surfaces:
        if not isinstance(surface, Metasurface):
            continue
        if not surface.use_metamodel:
            continue
        del surface.metamodel
        surface.propagator_cache = (None, None)
        if surface.use_circular_expansions:
            del surface.atom_1d.mmodel

    # Save the lens assembly with stdout and stderr suppressed.
    with utils.suppress_stdout_stderr():
        with open(save_path_pkl, "wb") as f:
            dill.dump(new_self, f)
        for i, surface in enumerate(lens_assembly.surfaces):
            if not isinstance(surface, Metasurface):
                continue
            if not surface.use_metamodel:
                continue
            surface.metamodel.save(
                f"surface_{i}_metamodel", save_path, overwrite
            )


def load_lens_assembly(
    name: str,
    save_dir: str = "./saved_lens_assemblies",
) -> LensAssembly:
    """Loads a lens assembly from disk.

    Args:
        name (str): the name of the lens assembly (folder name)
        save_dir (str, optional): The parent folder where the lens assembly is
            saved to. Defaults to "./saved_lens_assemblies".

    Returns:
        LensAssembly: The loaded lens assembly.
    """
    # Save the lens assembly with stdout and stderr suppressed.
    lens_assembly = dill.load(
        open(os.path.join(save_dir, name, "lens_assembly.pkl"), "rb")
    )
    for i, surface in enumerate(lens_assembly.surfaces):
        # Only apply to metasurfaces
        if not isinstance(surface, Metasurface):
            continue
        if not surface.use_metamodel:
            continue
        surface.metamodel = modeling.load_metamodel(
            "surface_{}_metamodel".format(i),
            save_dir=os.path.join(save_dir, name),
        )
        # Load the metamodel for the 1D atom if needed
        if surface.use_circular_expansions:
            surface.atom_1d.mmodel = surface.metamodel
    return lens_assembly


def optimize_single_lens_assembly(
    lens_assembly: LensAssembly,
    optimizer: tf.keras.optimizers.Optimizer,
    n_iter: int,
    verbose: int = 0,
    keep_best: bool = True,
) -> Tuple[LensAssembly, List[float]]:
    """Optimizes a single lens assembly.

    Args:
        lens_assembly: the lens assembly to optimize.
        optimizer: the optimizer to use.
        n_iter: the number of iterations to optimize.
        keep_best: whether to keep the best lens assembly.

    Returns:
        Tuple[LensAssembly, List[float]]: the optimized lens assembly and the
            history of the FOM.
    """
    variables = lens_assembly.get_variables()
    loss_history = []
    lowest_loss = np.inf
    best_lens_assembly_vars = lens_assembly.get_variables()
    if verbose <= 0:
        tr = range(n_iter)
    else:
        tr = tqdm.trange(n_iter, desc="Bar desc", leave=True)
    for _ in tr:
        with tf.GradientTape() as tape:
            loss = -lens_assembly.compute_FOM()
            loss += lens_assembly.compute_penalty()
        if keep_best:
            if loss < lowest_loss:
                lowest_loss = loss
                best_lens_assembly_vars = lens_assembly.get_variables().copy()
        grads = tape.gradient(loss, variables)
        for grad, variable in zip(grads, variables):
            grad = tf.math.real(grad)
            optimizer.apply_gradients([(grad, variable)])

        # record the loss
        loss_history.append(-loss.numpy())
        # update the progress bar
        if verbose > 0:
            tr.set_description(f"Loss: {loss.numpy():.6F}")
        lens_assembly.optimizer_hook()

    if keep_best:
        for variable_ts, best_variable_ts in zip(
            lens_assembly.get_variables(), best_lens_assembly_vars
        ):
            variable_ts.assign(best_variable_ts)
    return loss_history


def optimize_multiple_lens_assemblies(
    lens_assembly_arr: List[LensAssembly],
    optimizer: tf.keras.optimizers.Optimizer,
    n_iter: int,
    verbose: int = 0,
    keep_best: bool = True,
) -> Tuple[LensAssembly, List[float]]:
    """Optimizes multple lens assemblies.

    The gradient is accumulated across all lens assemblies siquentially.
    Then the graident is applied to all lens assemblies for each optimization
    iteration.

    Args:
        lens_assembly_arr: array of lens assembles to optimize.
        optimizer: the optimizer to use.
        n_iter: the number of iterations to optimize.
        verbose: the verbosity level.
        keep_best: whether to keep the best lens assembly.

    Returns:
        Tuple[LensAssembly, List[float]]: the optimized lens assembly and the
            history of the FOM.
    """

    variables = lens_assembly_arr[0].get_variables()
    # check that all lens assemblies have the same variables
    for lens_assembly in lens_assembly_arr:
        if not np.all(lens_assembly.get_variables() == variables):
            raise ValueError(
                "Not all lens assemblies have the same variables."
            )
    loss_history = []

    lowest_loss = np.inf
    best_lens_assembly_vars_list = [
        lens_assembly.get_variables() for lens_assembly in lens_assembly_arr
    ]

    # Create the progress bar
    if verbose <= 0:
        tr = range(n_iter)
    else:
        tr = tqdm.trange(n_iter, desc="Bar desc", leave=True)

    batch_grads = None
    for _ in tr:
        batch_loss = 0
        for lens_assembly in lens_assembly_arr:
            # calculate the loss for a single lens assembly
            with tf.GradientTape() as tape:
                single_loss = -lens_assembly.compute_FOM()
                single_loss += lens_assembly.compute_penalty()
            batch_loss += single_loss

            single_grads = tape.gradient(single_loss, variables)
            if batch_grads is None:
                batch_grads = single_grads
            else:
                batch_grads = [
                    batch_grad + single_grad
                    for batch_grad, single_grad in zip(
                        batch_grads, single_grads
                    )
                ]

            # record the loss
            lens_assembly.optimizer_hook()

        # Normalize the gradients
        batch_grads = [
            batch_grad / len(lens_assembly_arr) for batch_grad in batch_grads
        ]
        for grad, variable in zip(batch_grads, variables):
            grad = tf.math.real(grad)
            optimizer.apply_gradients([(grad, variable)])

        batch_loss /= len(lens_assembly_arr)
        loss_history.append(-batch_loss.numpy())

        # update the progress bar
        if verbose > 0:
            tr.set_description(f"Loss: {batch_loss.numpy():.6F}")

        # update the best lens assembly
        if keep_best:
            if batch_loss < lowest_loss:
                lowest_loss = batch_loss
                best_lens_assembly_vars_list = [
                    lens_assembly.get_variables().copy()
                    for lens_assembly in lens_assembly_arr
                ]

    # update the best lens assembly
    if keep_best:
        for lens_assembly, best_lens_assembly_vars in zip(
            lens_assembly_arr, best_lens_assembly_vars_list
        ):
            for variable_ts, best_variable_ts in zip(
                lens_assembly.get_variables(), best_lens_assembly_vars
            ):
                variable_ts.assign(best_variable_ts)

    return loss_history


def unbatch_incidence(
    incidence: Incidence,
) -> List[Incidence]:
    """Unbatches an incidence by the incident angles and wavelengths.

    Args:
        incidence: the incidence to unbatch.

    Returns:
        The unbacthed incidences.
    """
    # Get all permutations of the incident angles and wavelengths
    wavelength = copy.deepcopy(incidence.wavelength)
    theta = copy.deepcopy(incidence.theta)
    phi = copy.deepcopy(incidence.phi)
    permutations = list(itertools.product(wavelength, theta, phi))

    # Create the list of incidences to return
    incidences = []
    for i_wavelength, i_theta, i_phi in permutations:
        incidences.append(
            Incidence(
                wavelength=[i_wavelength],
                theta=[i_theta],
                phi=[i_phi],
            )
        )
    return incidences


def unbatch_lens_assembley(
    lens_assembly: LensAssembly,
) -> List[LensAssembly]:
    """Unbatches a lens assembly by the incident angles and wavelengths.

    Args:
        lens_assembly: the lens assembly to unbatch.

    Returns:
        The unbacthed lens assemblies.
    """
    # Get all permutations of the incident angles and wavelengths
    incidences = unbatch_incidence(lens_assembly.incidence)

    # Create the list of lens assemblies to return
    lens_assembly_arr = []
    for incidence in incidences:
        new_assembly = LensAssembly(
            surfaces=lens_assembly.surfaces,
            incidence=incidence,
            aperture_stop_index=lens_assembly.aperture_stop_index,
            figure_of_merit=lens_assembly.figure_of_merit,
            use_antialiasing=lens_assembly.use_antialiasing,
            use_padding=lens_assembly.use_padding,
        )
        lens_assembly_arr.append(new_assembly)
    return lens_assembly_arr


def structure_to_field_1d(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: Union[List[str], None] = None,
    use_padding: bool = True,
) -> propagation.Field1D:
    """"""
    if structure.use_mmodel:
        structure_to_field_method = structure_to_field_1d_mmodel
    else:
        structure_to_field_method = structure_to_field_1d_proto_unit_cell

    return structure_to_field_method(
        structure=structure,
        incidence=incidence,
        feature_order=feature_order,
        use_padding=use_padding,
    )


def structure_to_field_1d_proto_unit_cell(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: Union[List[str], None] = None,
    use_padding: bool = True,
) -> propagation.Field1D:
    """Converts a structure to a 1D field.

    Args:
        structure: the structure to convert.
        incidence: the incidence of the light.
        feature_order: unused.
        use_padding: whether to use padding for the field.

    Returns:
        The converted field.
    """
    structure_n_features = structure.tensor.shape[0]
    proto_uc_n_features = len(structure.proto_unit_cell.features)
    if structure_n_features != proto_uc_n_features:
        raise ValueError(
            "The number of features in the structure does not match the number of features in the metamodel."
        )

    fields_1d = rcwa.simulate_parameterized_unit_cells(
        parameter_tensor=structure.tensor,
        proto_cell=structure.proto_unit_cell,
        incidence=incidence,
        sim_config=structure.sim_config,
    )

    radius_size = fields_1d.shape[1]
    field_x = propagation.Field1D(
        tensor=fields_1d[..., 0],
        n_pixels=radius_size * 2,
        wavelength=incidence.wavelength,
        theta=incidence.theta,
        phi=incidence.phi,
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    field_y = propagation.Field1D(
        tensor=fields_1d[..., 1],
        n_pixels=radius_size * 2,
        wavelength=incidence.wavelength,
        theta=incidence.theta,
        phi=incidence.phi,
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    return field_x, field_y


def structure_to_field_1d_mmodel(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: Union[List[str], None] = None,
    use_padding: bool = True,
) -> propagation.Field1D:
    """Converts a structure to a 1D field.

    Args:
        structure: the structure to convert.
        incidence: the incidence of the light.
        mmodel: the metamodel to use for the conversion.
        feature_order: the order of the features in the structure
            tensor columns. The first feature has to be wavelength for
            chromatic optimizations. If None, the order of the features
            will be the same as the `features_attrs` in the metamodel.
        use_padding: whether to use padding for the field.

    Returns:
        The converted field.
    """

    structure_n_features = structure.tensor.shape[0]
    metamodel_n_features = len(structure.mmodel.protocell.features)
    if structure_n_features != metamodel_n_features:
        raise ValueError(
            "The number of features in the structure does not match the number of features in the metamodel."
        )

    # If no feature order is provided, use the order of the metamodel
    if feature_order is None:
        the_features = structure.mmodel.protocell.features.copy()
        feature_order = [a_feature.name for a_feature in the_features]
    else:
        feature_order = feature_order.copy()

    new_order = []
    for key in feature_order:
        # locate the index of the feature in the metamodel
        this_feature = next((x for x in the_features if x.name == key), None)
        new_order.append(the_features.index(this_feature))

    radius_size = structure.tensor.shape[-1]
    angles = len(incidence.theta) * len(incidence.phi)
    batch_number = len(incidence.wavelength) * angles

    # Repeat the lambda_base to complete the batch
    lambda_base = tf.cast(incidence.wavelength, tf.float32)
    wave_repeated = tf.repeat(lambda_base, radius_size)
    wave_angle_repeated = tf.repeat(wave_repeated, [angles])

    # tile the variables
    wave_angle_repeated = tf.expand_dims(wave_angle_repeated, axis=0)
    structure_var_tiled = tf.tile(structure.tensor, [1, batch_number])

    # join the inputs together
    inputs = tf.concat([wave_angle_repeated, structure_var_tiled], 0)
    # TODO: make the float position a parameter.
    inputs = tf.math.real(inputs)
    inputs = tf.cast(inputs, tf.float32)
    # transpose the inputs to match the model
    inputs = tf.transpose(inputs)
    outputs = structure.mmodel.model(inputs)
    # transpose back to the dim order
    outputs = tf.transpose(outputs)
    # avoid slicing, which kills the gradient
    x_vec = tf.cast([[1.0], [0.0]], tf.complex64)
    y_vec = tf.cast([[0.0], [1.0]], tf.complex64)
    tx = tf.reduce_sum(outputs * x_vec, axis=0)
    ty = tf.reduce_sum(outputs * y_vec, axis=0)
    # seperate the outputs into different wavelengths
    tx = tf.reshape(tx, [batch_number, radius_size])
    ty = tf.reshape(ty, [batch_number, radius_size])

    field_x = propagation.Field1D(
        tensor=tx,
        n_pixels=radius_size * 2,
        wavelength=incidence.wavelength,
        theta=incidence.theta,
        phi=incidence.phi,
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    field_y = propagation.Field1D(
        tensor=ty,
        n_pixels=radius_size * 2,
        wavelength=incidence.wavelength,
        theta=incidence.theta,
        phi=incidence.phi,
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    return field_x, field_y


def structure_to_field_2d(
    structure: AtomArray2D,
    incidence: Incidence,
    feature_order: Union[List[str], None] = None,
    use_padding: bool = True,
) -> propagation.Field2D:
    """Converts a structure to a 2D field.

    Args:
        structure: the structure to convert.
        incidence: the incidence of the light.
        mmodel: the metamodel to use for the conversion.
        feature_order: the order of the features in the structure
            tensor columns. The first feature has to be wavelength for
            chromatic optimizations. If None, the order of the features
            will be the same as the `features_attrs` in the metamodel.
        use_padding: whether to use padding for the field.

    Returns:
        The converted field.
    """

    dummy_field_x, dummy_field_y = structure_to_field_1d(
        structure=structure,
        incidence=incidence,
        feature_order=feature_order,
        use_padding=use_padding,
    )

    fields_rtn = []
    for dummy_field in [dummy_field_x, dummy_field_y]:
        dummy_tensor = dummy_field.tensor
        ts_shape = list(dummy_tensor.shape)
        n_pixels = int(np.sqrt(ts_shape[-1]))
        ts_shape.pop(-1)
        ts_shape.extend([n_pixels, n_pixels])
        dummy_tensor = tf.reshape(dummy_tensor, ts_shape)
        dummy_tensor = tf.cast(dummy_tensor, tf.complex64)

        fields_rtn.append(
            propagation.Field2D(
                tensor=dummy_tensor,
                n_pixels=n_pixels,
                wavelength=dummy_field.wavelength,
                theta=dummy_field.theta,
                phi=dummy_field.phi,
                period=structure.period,
                upsampling=1,
                use_padding=use_padding,
                use_antialiasing=True,
            )
        )

    return fields_rtn


def initialize_1d_atom_array_proto_unit_cell(
    n_pixels_radial: int,
    proto_unit_cell: rcwa.ProtoUnitCell,
    set_structures_variable: bool = False,
) -> AtomArray1D:
    """Initializes a 1D atom array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        proto_unit_cell: the proto unit cell to use for the initialization.
        set_structures_variable: whether to set the structure as a
            variable or not.

    Returns:
        The initialized atom array.
    """
    periodicity_xy = proto_unit_cell.proto_unit_cell.periodicity
    if periodicity_xy[0] != periodicity_xy[1]:
        raise ValueError(
            "The x and y periodicity of the unit cell must be equal."
            "Stay tuned for use of non-square unit cells for `Metasurface`."
        )

    variables = proto_unit_cell.generate_initial_variables(n_pixels_radial)
    if not set_structures_variable:
        variables = tf.constant(variables)

    return AtomArray1D(
        tensor=variables,
        period=periodicity_xy[0],
        proto_unit_cell=proto_unit_cell,
    )


def initialize_2d_atom_array_proto_unit_cell(
    n_pixels_radial: int,
    proto_unit_cell: rcwa.ProtoUnitCell,
    set_structures_variable: bool = False,
) -> AtomArray2D:
    """Initializes a 2D atom array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        proto_unit_cell: the proto unit cell to use for the initialization.
        set_structures_variable: whether to set the structure as a
            variable or not.

    Returns:
        The initialized atom array with shape (feature_0, feature_1, ..., n_pixels_x, n_pixels_y)
    """

    # Initialize the tensor
    dummy_atom_array = initialize_1d_atom_array_proto_unit_cell(
        n_pixels_radial=(n_pixels_radial * 2) ** 2,
        proto_unit_cell=proto_unit_cell,
        set_structures_variable=set_structures_variable,
    )

    return AtomArray2D(
        tensor=dummy_atom_array.tensor,
        period=proto_unit_cell.period,
        proto_unit_cell=proto_unit_cell,
    )


def initialize_1d_atom_array_metamodel(
    n_pixels_radial: int,
    period: float,
    mmodel: modeling.Metamodel,
    set_structures_variable: bool = False,
) -> AtomArray1D:
    """Initializes a 1D atom array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        period: the period of the structure in meters.
        mmmodel: the metamodel to use for the initialization.
        set_structures_variable: whether to set the structure as a
            variable or not.

    Returns:
        The initialized atom array.
    """

    # Initialize the tensor
    tensor_columns = []
    clip_value_min = []
    clip_value_max = []
    for feature in mmodel.protocell.features:
        vmin = [feature.vmin]
        vmax = [feature.vmax]
        clip_value_min.append(vmin)
        clip_value_max.append(vmax)
        tensor_columns.append(tf.random.uniform([n_pixels_radial], vmin, vmax))
    tensor = tf.stack(tensor_columns, axis=0)
    constraint_func = lambda x: tf.clip_by_value(
        x, clip_value_min, clip_value_max
    )

    if set_structures_variable:
        tensor = tf.Variable(tensor, constraint=constraint_func)

    return AtomArray1D(
        tensor=tensor,
        period=period,
        mmodel=mmodel,
    )


def initialize_1d_mask_array(
    n_pixels_radial: int,
    set_mask_variable: bool = False,
    init_bound: Tuple[float, float] = (0, 0),
) -> tf.Tensor:
    """Initializes a 1D mask array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        period: the period of the structure in meters.
        set_structures_variable: whether to set the structure as a
            variable or not.
        init_bound: the lower and upper bounds for the initialization.

    Returns:
        The initialized amplitude modulation coefficients.
    """

    tensor = tf.random.uniform([n_pixels_radial], init_bound[0], init_bound[1])
    constraint_func = lambda x: tf.clip_by_value(x, -1, 1)

    if set_mask_variable:
        tensor = tf.Variable(tensor, constraint=constraint_func)

    return tensor


def initialize_2d_mask_array(
    n_pixels_radial: int,
    set_structures_variable: bool = False,
) -> tf.Tensor:
    """Initializes a 2D atom array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        set_structures_variable: whether to set the structure as a
            variable or not.

    Returns:
        The initialized atom array.
    """

    # TODO: implement initialize_2d_atom_array_metamodel
    raise NotImplementedError()


def initialize_2d_atom_array_metamodel(
    n_pixels_radial: int,
    period: float,
    mmodel: modeling.Metamodel,
    set_structures_variable: bool = False,
    exclude_wavelength: bool = True,
) -> AtomArray2D:
    """Initializes a 2D atom array.

    Args:
        n_pixels_radial: the number of pixels in the radial direction.
        period: the period of the structure in meters.
        mmmodel: the metamodel to use for the initialization.
        set_structures_variable: whether to set the structure as a
            variable or not.
        exclude_wavelength: whether to exclude the wavelength from the
            feature initialization.

    Returns:
        The initialized atom array with shape (feature_0, feature_1, ..., n_pixels_x, n_pixels_y)
    """

    # Initialize the tensor
    dummy_atom_array = initialize_1d_atom_array_metamodel(
        n_pixels_radial=(n_pixels_radial * 2) ** 2,
        period=period,
        mmodel=mmodel,
        set_structures_variable=set_structures_variable,
    )

    return AtomArray2D(
        tensor=dummy_atom_array.tensor,
        period=period,
        mmodel=mmodel,
    )


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    from metabox import assembly, modeling, rcwa, utils

    cell_period = 442e-9

    radius = utils.Feature(vmin=0, vmax=221e-9, name="radius")
    circle_1 = rcwa.Circle(index=2, radius=radius)
    patterned_layer = rcwa.Layer(index=1, thickness=632e-9, shapes=[circle_1])
    substrate = rcwa.Layer(index=1.5, thickness=632e-9)
    cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate],
        periodicity=(cell_period, cell_period),
    )
    protocell = rcwa.ProtoUnitCell(cell)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=10e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        proto_unit_cell=protocell,
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
    )

    # Define the incidence wavelengths and angles.
    incidence = assembly.Incidence(
        wavelength=np.linspace(400e-9, 700e-9, 1),
        phi=[0],  # normal incidence
        theta=[0],  # normal incidence
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[metasurface],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,  # Define the figure of merit.
        use_x_pol=True,  # Use the x-polarization.
    )

    # Use the Adam optimizer to optimize the lens assembly. This rate should be
    # empirically determined.
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-8)
    optimizer.build(lens_assembly.get_variables())

    # Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
    history = assembly.optimize_single_lens_assembly(
        lens_assembly,
        optimizer,
        n_iter=20,
        verbose=1,
        keep_best=True,
    )
