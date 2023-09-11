import os, csv, tqdm, copy, dataclasses, gc, warnings, logging, glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Any, Dict, List, Tuple, Union

_ROOT = os.path.abspath(os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import interpolate

from metabox import raster, rcwa_tf, utils
from metabox.utils import CoordType, Feature, Incidence, ParameterType

# Suppress tensorflow warnings
tf.get_logger().setLevel(logging.ERROR)


def _get_features(parameter) -> List[Feature]:
    """Returns the features of the shape.

    Args:
        parameter: the parameter to get the features from.

    Yields:
        The features of the parameter."""
    if isinstance(parameter, Feature):
        yield parameter
    # recursively get features from nested iterables
    elif isinstance(parameter, (list, tuple, set)):
        for item in parameter:
            yield from _get_features(item)


class Parameterizable:
    """Defines a parameterizable object."""

    def __init__(self):
        self.unique_features = self.get_unique_features()

    def get_unique_features(self) -> List[Feature]:
        """Returns the unique features of the shape (non-recursively)."""
        return list(set(self.get_features()))

    def get_features(self) -> List[Feature]:
        """Returns the features of the shape."""
        for field in dataclasses.fields(self):
            parameter = getattr(self, field.name)
            yield from _get_features(parameter)

    def initialize_values(
        self,
        value_assignment: Union[
            None, Tuple[List[Feature], List[float]]
        ] = None,
    ) -> None:
        """Initializes the variables."""
        if value_assignment is not None:
            if len(value_assignment) != 2:
                raise ValueError(
                    "value_assignment must be a tuple of length 2."
                )
            elif len(value_assignment[0]) != len(value_assignment[1]):
                raise ValueError(
                    "value_assignment must be a tuple of lists of equal length."
                )

            # Assign the value to the field that contain the feature.
            features, values = value_assignment
            for feature, value in zip(features, values):
                # use value = None to initialize the feature randomly
                feature.initial_value = value

        for feature in self.unique_features:
            feature.initialize_value()

    def replace_feature_with_value(self) -> None:
        # set the field to the value
        for field in dataclasses.fields(self):
            parameter = getattr(self, field.name)
            if isinstance(parameter, Feature):
                setattr(self, field.name, parameter.value)
            elif isinstance(parameter, (list, tuple, set)):
                new_parameter = []
                for item in parameter:
                    if isinstance(item, Feature):
                        new_parameter.append(item.value)
                    else:
                        new_parameter.append(item)
                setattr(self, field.name, new_parameter)

    def get_variables(self) -> List[tf.Variable]:
        """Returns the variables of the shape."""
        variables = []
        for feature in self.unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables


@dataclasses.dataclass
class Shape(Parameterizable):
    """Defines a shape.

    Args:
        material: the `Material` or the ref. index of the shape.
    """

    material: Union[ParameterType, None]

    def __post_init__(self):
        return super().__init__()


@dataclasses.dataclass
class Polygon(Shape):
    """Defines a polygon.

    Args:
        material: the `Material` or the ref. index of the shape.
        vertices: the vertices of the polygon. List of (x, y) coordinates.
            Example_0: [(0, 0), (1, 0), (1, 1), (0, 1)]
            Example_1:
                var = Feature(vmin=0, vmax=1, name="var")
                [(0, 0), (var, 0), (var, 1), (0, 1)]
    """

    vertices: List[CoordType]

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        for vertex in self.vertices:
            if len(vertex) != 2:
                raise ValueError(
                    "Each vertex must be a tuple of (x, y) coordinates."
                )
        return super().__post_init__()

    def get_shape(self, wavelength: Union[float, None] = None):
        if type(self.material) is Material:
            if wavelength is None:
                raise ValueError(
                    "The wavelength must be given to rasterize a polygon with "
                    "a Material index."
                )
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Polygon(value=value, points=self.vertices)

    def get_vertices(self):
        """Returns the vertices of the polygon."""
        return self.vertices


@dataclasses.dataclass
class Rectangle(Shape):
    """Defines a rectangle.

    Args:
        material: the ref. index of the shape.
        x_width: the width of the rectangle in the x direction.
        y_width: the width of the rectangle in the y direction.
        x_pos: the x position of the rectangle. Default: 0
        y_pos: the y position of the rectangle. Default: 0
        rotation_deg: the rotation of the rectangle in degrees. Default: 0
        use_4_fold_symmetry: whether to use 4-fold symmetry. Default: False
            If True, the rectangle will be rotated by 0, 90, 180, 270 degrees.
            Then
    """

    x_width: ParameterType
    y_width: ParameterType
    x_pos: ParameterType = 0
    y_pos: ParameterType = 0
    rotation_deg: ParameterType = 0

    def __post_init__(self):
        return super().__post_init__()

    def get_shape(self, wavelength: Union[float, None] = None):
        if type(self.material) is Material:
            if wavelength is None:
                raise ValueError(
                    "The wavelength must be given to rasterize a rectangle with "
                    "a Material index."
                )
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Rectangle(
            value=value,
            center=(self.x_pos, self.y_pos),
            x_width=self.x_width,
            y_width=self.y_width,
            rotation_deg=self.rotation_deg,
        )

    def get_vertices(self):
        """Returns the vertices of the rectangle."""
        return raster.rectangle_to_vertices(
            center=(self.x_pos, self.y_pos),
            x_width=self.x_width,
            y_width=self.y_width,
            rotation_deg=self.rotation_deg,
        )


@dataclasses.dataclass
class Circle(Shape):
    """Defines a circle.

    Args:
        material: the ref. index of the shape.
        radius: the radius of the circle.
        center: the center of the circle. A tuple of (x, y) coordinates.
            Default: (0, 0)
    """

    radius: ParameterType
    x_pos: ParameterType = 0
    y_pos: ParameterType = 0

    def __post_init__(self):
        return super().__post_init__()

    def get_shape(self, wavelength: Union[float, None] = None):
        if type(self.material) is Material:
            if wavelength is None:
                raise ValueError(
                    "The wavelength must be given to rasterize a circle with "
                    "a Material index."
                )
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Circle(
            value=value,
            center=(self.x_pos, self.y_pos),
            radius=self.radius,
        )

    def get_vertices(self, num_of_vertices: int = 21):
        """Returns the vertices of the circle."""
        vertices = []
        for i in range(num_of_vertices):
            angle = i * 2 * np.pi / num_of_vertices
            x = self.x_pos + self.radius * np.cos(angle)
            y = self.y_pos + self.radius * np.sin(angle)
            vertices.append((x, y))
        return vertices


def duplicate_shape(shape: Shape, num_of_duplicates: int) -> List[Shape]:
    """Generates a list of duplicate parameterized shapes.

    The returned shapes share the same parameters as the input shape.
    But with different unique names.

    Args:
        shape: the shape to be duplicated.
        num_of_duplicates: the number of duplicates.
    Returns:
        A list of duplicated shapes.
    """
    shapes = []
    for i in range(num_of_duplicates):
        new_shape = copy.deepcopy(shape)
        features = new_shape.unique_features
        for feature in features:
            feature.name = f"{feature.name}~{i}"
        shapes.append(new_shape)
    return shapes


@dataclasses.dataclass
class Layer(Parameterizable):
    """Defines a layer.

    Args:
        material: the ref. index of the shape.
        thickness: the thickness of the layer in meters.
        shapes: the shapes of the layer. Tuple of Shape objects.
            Default: ()
    """

    material: Union[Feature, float, "Material"]
    thickness: Union[Feature, float]
    shapes: Tuple[Shape] = ()
    enforce_4fold_symmetry: bool = False

    def __post_init__(self):
        Parameterizable.__init__(self)

    def get_shapes(self, wavelength: Union[float, None] = None):
        """Returns the shapes of the layer."""
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.get_shape(wavelength))
        return shapes

    def initialize_values(
        self,
        value_assignment: Union[
            None, Tuple[List[Feature], List[float]]
        ] = None,
    ) -> None:
        """Initializes the layer variables."""
        super().initialize_values(value_assignment)
        for shape in self.shapes:
            shape.initialize_values(value_assignment)

    def get_layer_unique_features(self) -> List[Feature]:
        """Returns the unique features of the layer."""
        all_features = copy.deepcopy(self.unique_features)
        for shape in self.shapes:
            all_features.extend(shape.unique_features)
        return list(set(all_features))

    def get_variables(self) -> List[tf.Variable]:
        """Returns the variables of the layer."""
        all_features = copy.deepcopy(self.unique_features)
        for shape in self.shapes:
            all_features.extend(shape.unique_features)
        unique_features = list(set(all_features))
        variables = []
        for feature in unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables


@dataclasses.dataclass
class UnitCell(Parameterizable):
    """Defines a unit cell.

    Attributes:
        layers: the layers of the unit cell.
        periodicity: a tuple of (x, y) in meters that define the periodicity of
            the unit cell in the x and y direction.
        refl_index: the ref. index in the reflection region.
        tran_index: the ref. index in the transmission region.
    """

    layers: List[Layer]
    periodicity: Tuple[ParameterType, ParameterType]
    refl_index: ParameterType = 1.0
    tran_index: ParameterType = 1.0

    def __post_init__(self):
        super().__init__()
        if len(self.periodicity) != 2:
            raise ValueError(
                "The periodicity must be a tuple of (x, y) in meters."
            )

    def initialize_values(
        self,
        value_assignment: Union[
            None, Tuple[List[Feature], List[float]]
        ] = None,
    ) -> None:
        """Initializes the layer variables."""
        super().initialize_values(value_assignment)
        for layer in self.layers:
            layer.initialize_values(value_assignment)

    def replace_features(self):
        """Replaces the features with the given values."""
        _replace_feature_with_value_in_dataclass(self)

    def get_cell_unique_features(self) -> List[Feature]:
        """Returns the unique features of the unit cell."""
        all_features = copy.deepcopy(self.unique_features)
        for layer in self.layers:
            all_features.extend(layer.get_layer_unique_features())
        return list(set(all_features))

    def get_variables(self) -> List[tf.Variable]:
        unique_features = self.get_cell_unique_features()
        variables = []
        for feature in unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables

    def get_epsilon(self, x_resolution: int, wavelength: float) -> tf.Tensor:
        """Returns the permittivity of the unit cell.

        Args:
            x_resolution: the resolution of the permittivity in the x direction.

        Returns:
            The permittivity of the unit cell as a tf.Tensor.
        """
        pixel_density = self.periodicity[0] / float(x_resolution)
        epsilon_all = []
        for layer in self.layers:
            epsilon_layer = (
                _rasterize_layer(
                    layer=layer,
                    periodicity=self.periodicity,
                    pixel_density=pixel_density,
                    enforce_4fold_symmetry=layer.enforce_4fold_symmetry,
                    wavelength=wavelength,
                )
                ** 2
            )
            epsilon_all.append(tf.cast(epsilon_layer, tf.complex64))
        epsilon_all = tf.cast(epsilon_all, tf.complex64)
        return tf.stack(epsilon_all, axis=0)

    def get_thickness(self) -> tf.Tensor:
        """Returns the thickness of the unit cell as a tf.Tensor."""
        return tf.cast(
            [tf.math.real(layer.thickness) for layer in self.layers],
            tf.float32,
        )

    def find_feature_index(self, feature_str):
        """Returns the index of the feature with the given name."""
        for i, feature in enumerate(self.unique_features):
            if feature.name == feature_str:
                return i
        raise ValueError("Feature not found.")


def _replace_this_feature_with_value_recursively(
    parent: Any, child_field: Any
) -> None:
    """Recursively replaces the features with their values.

    if the field is a feature, replace it with its value. If the field is a
    list, tuple, or set, recursively call this function on each element of the
    list, tuple, or set. If the field is a dataclass, recursively call this
    function on each field of the dataclass.

    Args:
        parent: the parent of the field.
        child_field: the field to replace.
    """
    parent_is_list_or_tuple = isinstance(parent, (list, tuple))
    parent_is_tuple = isinstance(parent, tuple)
    if parent_is_list_or_tuple:
        field_content = child_field
    else:
        field_content = getattr(parent, child_field.name)

    if isinstance(field_content, Feature):
        if field_content.value is None:
            field_content.initialize_value()
        if parent_is_list_or_tuple:
            if parent_is_tuple:
                parent = list(parent)
            idx = parent.index(child_field)
            parent[idx] = field_content.value
            if parent_is_tuple:
                parent = tuple(parent)
        else:
            setattr(parent, child_field.name, field_content.value)
    elif isinstance(field_content, (list, tuple)):
        for child_field_child_field in field_content:
            _replace_this_feature_with_value_recursively(
                parent=field_content, child_field=child_field_child_field
            )
    elif dataclasses.is_dataclass(field_content):
        _replace_feature_with_value_in_dataclass(field_content)
    # do nothing for the other cases


def _replace_feature_with_value_in_dataclass(dataclass_instance) -> None:
    """Replaces the features with their values in a dataclass instance.

    Args:
        dataclass_instance: the dataclass instance to replace the features in.
    """
    # set the field to the value
    for field in dataclasses.fields(dataclass_instance):
        _replace_this_feature_with_value_recursively(dataclass_instance, field)


@dataclasses.dataclass
class ProtoUnitCell:
    """Defines an archetype of UnitCell (i.e. parameterized by `Feature`s).

    Provides an interface to generate an array of unit cells from a tensor with
    shape (n_features, n_unit_cells).

    Attributes:
        proto_unit_cell: the unit cell that the children units cells are
            based of. The children unit cells will share the same Features
            as the parent unit cell.
    """

    proto_unit_cell: UnitCell

    def __post_init__(self):
        self.features = self.proto_unit_cell.get_cell_unique_features()

    def generate_initial_variables(self, n_cells: int) -> tf.Tensor:
        """Returns a tensor of initial variable parameters for the unit cells.

        The tensor has shape (n_features, n_unit_cells).

        Args:
            n_cell: the number of unit cells to generate.

        Returns:
            A tensor of initial parameters for the unit cells.
        """
        # Initialize the tensor
        tensor_columns = []
        clip_value_min = []
        clip_value_max = []
        for feature in self.proto_unit_cell.get_cell_unique_features():
            vmin = [feature.vmin]
            vmax = [feature.vmax]
            clip_value_min.append(vmin)
            clip_value_max.append(vmax)
            tensor_columns.append(tf.random.uniform([n_cells], vmin, vmax))

        tensor = tf.stack(tensor_columns, axis=0)
        constraint_func = lambda x: tf.clip_by_value(
            x, clip_value_min, clip_value_max
        )
        return tf.Variable(tensor, constraint=constraint_func)

    def generate_cells_from_parameter_tensor(
        self, tensor: tf.Tensor
    ) -> List[UnitCell]:
        """Returns an array of unit cells from a tensor shape: (n_cell, n_feat).

        Args:
            tensor(tf.Tensor): a tensor with shape (n_unit_cells, n_features).

        Raises:
            ValueError: when the tensor does not have the correct shape.
        """
        if tensor.shape[0] != len(self.features):
            raise ValueError(
                "The tensor must have shape (n_features, n_unit_cells)."
            )

        unit_cell_array = []
        for i in range(tensor.shape[-1]):
            unit_cell = copy.deepcopy(self.proto_unit_cell)
            features = unit_cell.get_cell_unique_features()
            parameters = tensor[:, i]
            for feature, init_value in zip(features, parameters):
                feature.set_value(init_value)
            unit_cell.replace_features()
            unit_cell_array.append(unit_cell)
        return unit_cell_array


def _rasterize_layer(
    layer: Layer,
    periodicity: Tuple[ParameterType, ParameterType],
    pixel_density: float,
    enforce_4fold_symmetry: bool = False,
    wavelength: Union[float, None] = None,
) -> raster.Canvas:
    """Rasterizes a layer.

    Args:
        layer: the layer to rasterize.
        periodicity: a tuple of (x, y) in meters that define the periodicity of
            the unit cell in the x and y direction.
        pixel_density: the pixel density in meters.
        enforce_4fold_symmetry: whether to make the layer 4-fold symmetric.
            If true, then the layer will be mirrored along the x and y axes,
            then added to its transpose.
        wavelength: the wavelength of the simulation in meters.

    Returns:
        The rasterized layer.
    """
    if type(layer.material) is Material:
        if wavelength is None:
            raise ValueError(
                "The wavelength must be given to rasterize a layer with "
                "a Material index."
            )
        layer_value = layer.material.index_at(wavelength)
    else:
        layer_value = layer.material

    return raster.Canvas(
        x_width=periodicity[0],
        y_width=periodicity[1],
        spacing=pixel_density,
        background_value=layer_value,
        enforce_4fold_symmetry=enforce_4fold_symmetry,
    ).rasterize(layer.get_shapes(wavelength))


def get_avaliable_materials(
    custom_csv_dir: Union[str, None] = None
) -> List[str]:
    """Returns a list of avaliable material strings in the given directory."""

    if custom_csv_dir is None:
        custom_csv_dir = os.path.join(_ROOT, "material_data")

    avail_materials_dir = glob.glob(os.path.join(custom_csv_dir, "*.csv"))
    avali_materials = [
        os.path.split(file_path)[-1].split(".")[0]
        for file_path in avail_materials_dir
    ]
    return avali_materials


@dataclasses.dataclass
class Material:
    """Defines a material class.

    A material provides a way to define the refractive index of a material
    given the wavelength of the simulation.

    Attributes:
        name: the name of the material. The .csv file name must be [name].csv
            The file extension i.e. `.csv` must be in lowercase.
        custom_csv_dir: the path to the csv file folder that contains the refractive
            index data. The data can be downloaded from refractiveindex.info.
            Just search for the material and download the csv file, under the
            "Data" section. Save the [CSV - comma separated] file as `csv_file_dir`.
    """

    name: str
    custom_csv_dir: Union[str, None] = None

    def __post_init__(self):
        if self.custom_csv_dir is None:
            self.custom_csv_dir = os.path.join(_ROOT, "material_data")

        csv_dir = os.path.join(self.custom_csv_dir, self.name + ".csv")
        if not os.path.exists(csv_dir):
            avaliable_materials = get_avaliable_materials(self.custom_csv_dir)
            default_mat_str = ", ".join(avaliable_materials)
            raise ValueError(
                f"The csv file for {self.name} does not exist in {self.custom_csv_dir}.\n"
                f"Could not find {csv_dir}\n"
                f"Avaliable materials: {default_mat_str}"
            )

        self.wl_n = []
        self.wl_k = []
        self.n = []
        self.k = []

        with open(csv_dir, "r") as file:
            reader = csv.reader(file)
            mode = None
            for row in reader:
                if len(row) == 0 or row[0].strip() == "":
                    continue  # Skip empty rows

                if row[1] == "n":
                    mode = "n"
                    continue
                elif row[1] == "k":
                    mode = "k"
                    continue

                if mode == "n":
                    self.wl_n.append(float(row[0]) * 1e-6)
                    self.n.append(float(row[1]))
                elif mode == "k":
                    self.wl_k.append(float(row[0]) * 1e-6)
                    self.k.append(float(row[1]))

        self.min_wl_n = min(self.wl_n)
        self.max_wl_n = max(self.wl_n)
        if len(self.wl_k) > 0:
            self.min_wl_k = min(self.wl_k)
            self.max_wl_k = max(self.wl_k)

        self.n_interp = interpolate.interp1d(self.wl_n, self.n)

        if len(self.wl_k) > 0:
            self.k_interp = interpolate.interp1d(self.wl_k, self.k)

    def index_at(self, wavelength):
        """Returns the refractive index at the given wavelength."""
        if not (self.min_wl_n <= wavelength <= self.max_wl_n):
            raise ValueError(f"Wavelength {wavelength} is out of range.")

        n_value = self.n_interp(wavelength)
        if hasattr(self, "k_interp") and self.k_interp is not None:
            k_value = self.k_interp(wavelength)
        else:
            k_value = 0j

        return n_value + 1.0j * k_value


@dataclasses.dataclass
class SimConfig:
    """Defines a simulation configuration.

    The SimConfig class is an immutable dataclass that provides the configuration
    and precomuputed data for the RCWA simulation.
    TODO: add fast convolution matrix

    Attributes:
        xy_harmonics: a tuple of (x, y) positive odd ints of Fourier harmonics.
        x_resolution: the grid x resolution of the simulation of the real space.
            Note that the grid y resolution is determined by the aspect ratio of
            the unit cell.
        minibatch_size: the minibatch size of the simulation.
        return_tensor: whether to use tensor as the output.
            If False, then the following arguments are ignored:
                return_zeroth_order, use_transmission, include_tz.
                And the output is the in the form of `SimResult`.
            If True, the output is in the form of a tf.Tensor, and the following
                arguments are used:
                return_zeroth_order=True, use_transmission=True, include_z_comp=False.
        return_zeroth_order: whether to use zeroth order diffraction as the output.
        use_transmission: whether to use transmission as the output.
        include_z: whether to include the z component of the electric field.
    """

    xy_harmonics: Tuple[int, int]
    resolution: int
    minibatch_size: int = 100
    return_tensor: bool = False
    return_zeroth_order: Union[bool, None] = None
    use_transmission: Union[bool, None] = None
    include_z_comp: Union[bool, None] = None

    def __post_init__(self):
        if self.xy_harmonics[0] % 2 != 1 or self.xy_harmonics[0] < 1:
            raise ValueError("xy_harmonics[0] must be a positive odd int.")
        elif self.xy_harmonics[1] % 2 != 1 or self.xy_harmonics[1] < 1:
            raise ValueError("xy_harmonics[1] must be a positive odd int.")

        if self.return_tensor:
            if self.return_zeroth_order is None:
                self.return_zeroth_order = True
            if self.use_transmission is None:
                self.use_transmission = True
            if self.include_z_comp is None:
                self.include_z_comp = False
        else:
            if (
                (self.return_zeroth_order is not None)
                or (self.use_transmission is not None)
                or (self.include_z_comp is not None)
            ):
                warnings.warn(
                    "When return_tensor is False, the following arguments are "
                    "ignored: return_zeroth_order, use_transmission, include_z_comp."
                )


@dataclasses.dataclass
class SimInstance:
    """Defines a simulation instance.

    Attributes:
        unit_cell_array: an array of unit cells to be simulated.
        incidence: the incidence of the simulation.
        sim_config: the simulation configuration.
    """

    unit_cell_array: List[UnitCell]
    incidence: Incidence
    sim_config: SimConfig

    def __post_init__(self):
        # check all x and y periods are the same
        x_periodicitys = [
            unit_cell.periodicity[0] for unit_cell in self.unit_cell_array
        ]
        y_periodicitys = [
            unit_cell.periodicity[1] for unit_cell in self.unit_cell_array
        ]
        if len(set(x_periodicitys)) != 1:
            raise ValueError("All x periods must be the same.")
        if len(set(y_periodicitys)) != 1:
            raise ValueError("All y periods must be the same.")

        # check all ref. indices are the same for transmission and reflection regions
        refl_indices = [
            unit_cell.refl_index for unit_cell in self.unit_cell_array
        ]
        tran_indices = [
            unit_cell.tran_index for unit_cell in self.unit_cell_array
        ]
        if len(set(refl_indices)) != 1:
            raise ValueError(
                "All ref. indices must be the same for reflection region."
            )
        if len(set(tran_indices)) != 1:
            raise ValueError(
                "All ref. indices must be the same for transmission region."
            )

    def get_variables(self) -> List[tf.Variable]:
        """Returns the variables of the simulation instance."""
        variables = []
        for unit_cell in self.unit_cell_array:
            variables.extend(unit_cell.get_variables())
        return variables


@dataclasses.dataclass
class SimResult:
    """The result of an RCWA simulation.

    Attributes:
        rx: the x component of the reflected diffraction coeff.
        ry: the y component of the reflected diffraction coeff.
        rz: the z component of the reflected diffraction coeff.
        r_eff: the reflective efficiency.
        r_power: the total reflected power.
        tx: the x component of the transmitted diffraction coeff.
        ty: the y component of the transmitted diffraction coeff.
        tz: the z component of the transmitted diffraction coeff.
        t_eff: the transmissive efficiency.
        t_power: the total transmitted power.
    """

    rx: tf.Tensor
    ry: tf.Tensor
    rz: tf.Tensor
    r_eff: tf.Tensor
    r_power: tf.Tensor
    tx: tf.Tensor
    ty: tf.Tensor
    tz: tf.Tensor
    t_eff: tf.Tensor
    t_power: tf.Tensor
    xy_harmonics: Tuple[int, int]

    @staticmethod
    def _get_0th(fields, xy_harmonics):
        return fields[:, :, 0, np.prod(xy_harmonics) // 2, 0]

    def ref_field(self, config: SimConfig) -> tf.Tensor:
        """Returns the reflected diffraction coefficients.

        Returns:
            The reflected field according to the simulation configuation.
        """
        if config.return_zeroth_order:
            rx = self._get_0th(self.rx, self.xy_harmonics)
            ry = self._get_0th(self.ry, self.xy_harmonics)
            rz = self._get_0th(self.rz, self.xy_harmonics)
        else:
            rx, ry, rz = self.rx, self.ry, self.rz

        if config.include_z_comp:
            return tf.stack([rx, ry, rz], axis=-1)
        else:
            return tf.stack([rx, ry], axis=-1)

    def trn_field(self, config: SimConfig) -> tf.Tensor:
        """Returns the transmitted diffraction coefficients.

        Args:
            config: the simulation configuration.
        Returns:
            The transmitted field according to the simulation configuation.
        """
        if config.return_zeroth_order:
            tx = self._get_0th(self.tx, self.xy_harmonics)
            ty = self._get_0th(self.ty, self.xy_harmonics)
            tz = self._get_0th(self.tz, self.xy_harmonics)
        else:
            tx, ty, tz = self.tx, self.ty, self.tz

        if config.include_z_comp:
            return tf.stack([tx, ty, tz], axis=-1)
        else:
            return tf.stack([tx, ty], axis=-1)

    def get_result_using_config(
        self, config: SimConfig
    ) -> Union["SimResult", tf.Tensor]:
        """Returns the result according to the simulation configuation.

        Args:
            config: the simulation configuration.
        Returns:
            The result according to the simulation configuation.
        """
        if not config.return_tensor:
            return self

        if config.use_transmission:
            return self.trn_field(config)
        else:
            return self.ref_field(config)


def minibatch_sim_instance(
    sim_instance: SimInstance, minibatch_size: int
) -> List[SimInstance]:
    """Generates a list of minibatch simulation instances.

    Args:
        sim_instance: the simulation instance.
        batch_size: the batch size.
    Returns:
        A list of minibatch simulation instances.
    """
    unit_cell_array_chunks = [
        sim_instance.unit_cell_array[i : i + minibatch_size]
        for i in range(0, len(sim_instance.unit_cell_array), minibatch_size)
    ]
    sim_instance_array = []
    for unit_cell_array in unit_cell_array_chunks:
        sim_instance_array.append(
            SimInstance(
                unit_cell_array=unit_cell_array,
                incidence=sim_instance.incidence,
                sim_config=sim_instance.sim_config,
            )
        )
    return sim_instance_array


def combine_sim_results(
    sim_results: Union[List[SimResult], List[tf.Tensor]]
) -> Union[SimResult, tf.Tensor]:
    """Combines a list of simulation results into one

    Args:
        sim_results: the list of simulation results.
    Returns:
        The combined simulation result.
    """
    if isinstance(sim_results[0], tf.Tensor):
        return tf.concat(sim_results, axis=1)

    rx = tf.concat([sim_result.rx for sim_result in sim_results], axis=1)
    ry = tf.concat([sim_result.ry for sim_result in sim_results], axis=1)
    rz = tf.concat([sim_result.rz for sim_result in sim_results], axis=1)
    r_eff = tf.concat([sim_result.r_eff for sim_result in sim_results], axis=1)
    r_power = tf.concat(
        [sim_result.r_power for sim_result in sim_results], axis=1
    )
    tx = tf.concat([sim_result.tx for sim_result in sim_results], axis=1)
    ty = tf.concat([sim_result.ty for sim_result in sim_results], axis=1)
    tz = tf.concat([sim_result.tz for sim_result in sim_results], axis=1)
    t_eff = tf.concat([sim_result.t_eff for sim_result in sim_results], axis=1)
    t_power = tf.concat(
        [sim_result.t_power for sim_result in sim_results], axis=1
    )
    xy_harmonics = sim_results[0].xy_harmonics
    return SimResult(
        rx=rx,
        ry=ry,
        rz=rz,
        r_eff=r_eff,
        r_power=r_power,
        tx=tx,
        ty=ty,
        tz=tz,
        t_eff=t_eff,
        t_power=t_power,
        xy_harmonics=xy_harmonics,
    )


def simulate_parameterized_unit_cells(
    parameter_tensor: tf.Tensor,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
) -> tf.Tensor:
    """Simulate RCWA and precompute the JVP with better memory efficiency.

    This method computes the zeroth order diffraction coefficients and the
    Jacobian of the diffraction coefficients with respect to the unit cell
    parameters.

    Args:
        parameter_tensor: the tensor of unit cell parameters, in the shape
            (num_features, num_unit_cells).
        proto_cell: a parameterized unit cell.
        incidence: the incidence data.
        sim_config: the simulation configuration.

    Returns:
        The 0th order diffraction coefficients. In the form of a tf.Tensor of
        shape (batch_size, num_unit_cells, 2).
    """

    minibatch_size = sim_config.minibatch_size
    # fwd_jvp_mode: whether to use the forward JVP method to compute gradient.
    # If False, the gradient will be computed using the reverse VJP method.
    fwd_jvp_mode = False
    # enable_jac_cache: uses custom gradient method to greatly reduce memory
    # usage for large number of minibatches.
    enable_jac_cache = False

    if enable_jac_cache:
        simulate_func = simulate_parameterized_unit_cells_one_batch
    else:
        simulate_func = simulate_parameterized_unit_cells_one_batch_no_jvp

    if minibatch_size < parameter_tensor.shape[1]:
        # TODO: multi-GPU support
        parameters_chunks = [
            parameter_tensor[:, i : i + minibatch_size]
            for i in range(0, parameter_tensor.shape[1], minibatch_size)
        ]
        sim_results = []
        for parameters_chunk in tqdm.tqdm(parameters_chunks):
            sim_results.append(
                simulate_func(
                    parameter_tensor=parameters_chunk,
                    proto_cell=proto_cell,
                    incidence=incidence,
                    sim_config=sim_config,
                    fwd_jvp_mode=fwd_jvp_mode,
                )
            )
            gc.collect()
        return combine_sim_results(sim_results)

    return simulate_func(
        parameter_tensor=parameter_tensor,
        proto_cell=proto_cell,
        incidence=incidence,
        sim_config=sim_config,
        fwd_jvp_mode=fwd_jvp_mode,
    )


def simulate_parameterized_unit_cells_one_batch(
    parameter_tensor: tf.Tensor,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
    fwd_jvp_mode: bool = True,
) -> tf.Tensor:
    if not sim_config.return_tensor:
        raise ValueError(
            "SimConfig.return_tensor=True is required for this method."
        )

    if len(proto_cell.features) == 0:
        raise ValueError("The proto cell has no features (not parameterized).")

    # use nested inner functions to avoid passing sim_instance as an argument
    # tf.custom_gradient will otherwise blindly convert every argument to a
    # tf.Tensor on the graph.
    @tf.custom_gradient
    def inner_simulate(inner_parameter_tensor):
        nonlocal proto_cell, incidence, sim_config, fwd_jvp_mode
        output, jvp = _compute_output_and_jvp(
            inner_parameter_tensor.numpy(),
            proto_cell,
            incidence,
            sim_config,
            fwd_jvp_mode,
        )

        def backward(upstream):
            # save the gradient of the output with respect to the variables
            nonlocal jvp
            # jvp shape: (batch_size, n_cells, t_xy, n_features)
            # upstream shape: (batch_size, n_cells, t_xy)
            # downstream shape: (n_features, n_cells)
            # upstream shape now: (batch_size, n_cells, t_xy, 1)
            upstream = upstream[..., tf.newaxis]
            jvp = tf.cast(jvp, upstream.dtype)
            jvp = jvp * upstream
            # now jvp shape: (n_cells, t_xy, n_features)
            jvp = tf.reduce_sum(jvp, axis=0, keepdims=False)
            # now jvp shape: (n_cells, n_features)
            jvp = tf.reduce_sum(jvp, axis=1, keepdims=False)
            # transpose to match the shape of the input
            return tf.transpose(jvp)

        return output, backward

    return inner_simulate(parameter_tensor)


def simulate_parameterized_unit_cells_one_batch_no_jvp(
    parameter_tensor: tf.Tensor,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
    fwd_jvp_mode: bool = False,
) -> tf.Tensor:
    if not sim_config.return_tensor:
        raise ValueError(
            "SimConfig.return_tensor=True is required for this method."
        )

    if len(proto_cell.features) == 0:
        raise ValueError("The proto cell has no features (not parameterized).")

    children = proto_cell.generate_cells_from_parameter_tensor(
        parameter_tensor
    )
    sim_instance = SimInstance(
        unit_cell_array=children,
        incidence=incidence,
        sim_config=sim_config,
    )
    return simulate_one(sim_instance)


def _compute_output_and_jvp(
    parameter_tensor: np.ndarray,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
    fwd_jvp_mode: bool = True,
    fast_mode: bool = True,
):
    """Compute the output and JVP of the output w.r.t. the parameters."""
    if not sim_config.return_zeroth_order:
        raise NotImplementedError(
            "Calculation of Jacobian is only supported when"
            "SimConfig.return_zeroth_order=True."
        )

    parameter_tensor = tf.convert_to_tensor(parameter_tensor)

    if fwd_jvp_mode:
        # the vector used as the column in the forward gradient accumulation
        n_features = parameter_tensor.shape[0]
        n_cells = parameter_tensor.shape[1]
        tangent_array = []
        for i in range(n_features):
            new_zero = np.zeros((n_features, 1))
            new_zero[i, :] = 1
            tangent_array.append(np.ones((n_features, n_cells)) * new_zero)

        jvps = []
        for tangent in tangent_array:
            with tf.autodiff.ForwardAccumulator(
                primals=parameter_tensor,  # Evaluation point
                tangents=tangent,  # Tangent vector
            ) as acc:
                children = proto_cell.generate_cells_from_parameter_tensor(
                    parameter_tensor
                )
                sim_instance = SimInstance(
                    unit_cell_array=children,
                    incidence=incidence,
                    sim_config=sim_config,
                )
                # output shape: (batch_size, n_cells, t_xy)
                output = simulate_one(sim_instance)
            # in the shape of (batch_size, n_cells, t_xy)
            jvp = acc.jvp(output)
            jvps.append(jvp)
        # in the shape of (batch_size, n_cells, t_xy, n_features)
        jac = np.stack(jvps, axis=-1)
        jac = tf.math.conj(jac)

    else:
        cell_param_list = []
        # parameter_tensor shape: (n_features, n_cells)
        for i in range(parameter_tensor.shape[1]):
            cell_param_list.append(parameter_tensor[:, i])
        parameter_tensor = tf.convert_to_tensor(parameter_tensor)
        with tf.GradientTape(persistent=True) as tape:
            for i in range(len(cell_param_list)):
                cell_param = cell_param_list[i]
                # cell_param shape now: (n_features, 1)
                cell_param = tf.convert_to_tensor(cell_param)[..., tf.newaxis]
                tape.watch(cell_param)
                cell_param_list[i] = cell_param
            parameter_tensor = tf.concat(cell_param_list, axis=-1)
            children = proto_cell.generate_cells_from_parameter_tensor(
                parameter_tensor
            )
            sim_instance = SimInstance(
                unit_cell_array=children,
                incidence=incidence,
                sim_config=sim_config,
            )
            # output in the shape of (batch_size, n_cells, t_xy)
            output = simulate_one(sim_instance)
            outputs = tf.split(
                output, num_or_size_splits=len(cell_param_list), axis=1
            )

        jac_array = []
        for cell_output, cell_param in zip(outputs, cell_param_list):
            # jac_cell shape: (batch_size, 1, t_xy, n_features, 1)
            jac_cell = tape.jacobian(cell_output, cell_param)
            # shape now: (batch_size, t_xy, n_features, 1)
            # jac_cell = tf.math.conj(jac_cell)[:, 0, ...]
            jac_cell = jac_cell[:, 0, ...]
            jac_array.append(jac_cell)
        del tape
        # jac shape: (batch_size, t_xy, n_features, n_cells)
        jac = tf.concat(jac_array, axis=-1)
        # final jac in the shape of (batch_size, n_cells, t_xy, n_features)
        jac = tf.transpose(jac, perm=[0, 3, 1, 2])

    return output, jac


def simulate(sim_instance: SimInstance) -> SimResult:
    """Simulates the periodic unit cell using RCWA.

    Calculates the transmission/reflection coefficients for a unit cell with a
    given a simulation instance (SimInstance), which contains the unit cell,
    incidence, and simulation configuration.

    Args:
        sim_instance: the simulation instance.
    Returns:
        The simulation result.
    """
    minibatch_size = sim_instance.sim_config.minibatch_size
    if minibatch_size < len(sim_instance.unit_cell_array):
        sim_instances = minibatch_sim_instance(
            sim_instance=sim_instance, minibatch_size=minibatch_size
        )
        sim_results = simulate_batch(sim_instances=sim_instances)
        return combine_sim_results(sim_results=sim_results)
    else:
        return simulate_one(sim_instance=sim_instance)


def simulate_batch(sim_instances: List[SimInstance]) -> List[SimResult]:
    """Simulates a batch of periodic unit cells using RCWA.

    Calculates the transmission/reflection coefficients for a batch of unit
    cells with a given a list of simulation instances (SimInstance), which
    contains the unit cell, incidence, and simulation configuration.

    TODO: parallelize this function.

    Args:
        sim_instances: list of simulation instances.
    Returns:
        The list of simulation results.
    """
    return [simulate_one(sim_instance) for sim_instance in sim_instances]


def simulate_one(sim_instance: "SimInstance") -> "SimResult":
    """Simulates the periodic unit cell using RCWA.

    Calculates the transmission/reflection coefficients for a unit cell with a
    given a simulation instance (SimInstance), which contains the unit cell,
    incidence, and simulation configuration.

    Args:
        sim_instance: the simulation instance.
    Returns:
        The simulation result.
    """

    incidence_dict = utils.unravel_incidence(sim_instance.incidence)

    batched_layer_thicknesses = []
    for unit_cell in sim_instance.unit_cell_array:
        layer_thicknesses = unit_cell.get_thickness()
        layer_thicknesses = tf.cast(layer_thicknesses, tf.complex64)
        layer_thicknesses = layer_thicknesses[
            tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis
        ]
        batched_layer_thicknesses.append(layer_thicknesses)
    layer_thicknesses = tf.concat(batched_layer_thicknesses, axis=1)

    # permittivity in reflection region
    er1 = sim_instance.unit_cell_array[0].refl_index ** 2
    # permittivity in transmission region
    er2 = sim_instance.unit_cell_array[0].tran_index ** 2

    ### Step 2: Compute permittivity and permeability ###
    n_cells = len(sim_instance.unit_cell_array)
    batch_size = len(incidence_dict["wavelength"])
    ER_t_arrary = []
    for unit_cell in sim_instance.unit_cell_array:
        ER_t_wl = []
        for wavelength in incidence_dict["wavelength"]:
            this_epsilon = unit_cell.get_epsilon(
                sim_instance.sim_config.resolution,
                wavelength=wavelength,
            )
            this_epsilon = this_epsilon[
                tf.newaxis, tf.newaxis, tf.newaxis, :, :, :
            ]
            ER_t_wl.append(this_epsilon)
        ER_t = tf.concat(ER_t_wl, axis=0)
        ER_t_arrary.append(ER_t)
    ER_t = tf.concat(ER_t_arrary, axis=1)
    # Dielectric materials for now
    UR_t = tf.ones_like(ER_t)

    refl_n = sim_instance.unit_cell_array[0].refl_index

    output = rcwa_tf.simulate_rcwa(
        incidence_dict,
        PQ=sim_instance.sim_config.xy_harmonics,
        n_cells=n_cells,
        n_layers=len(sim_instance.unit_cell_array[0].layers),
        layer_thicknesses=layer_thicknesses,
        L_xy=sim_instance.unit_cell_array[0].periodicity,
        er1=er1,
        er2=er2,
        ER_t=ER_t,
        UR_t=UR_t,
        refl_n=refl_n,
    )

    # Store the transmission/reflection coefficients and powers in a SimResult.
    result = SimResult(
        rx=tf.math.conj(output["rx"]),
        ry=tf.math.conj(output["ry"]),
        rz=tf.math.conj(output["rz"]),
        r_eff=tf.math.conj(output["R"]),
        r_power=tf.math.conj(output["REF"]),
        tx=tf.math.conj(output["tx"]),
        ty=tf.math.conj(output["ty"]),
        tz=tf.math.conj(output["tz"]),
        t_eff=tf.math.conj(output["T"]),
        t_power=tf.math.conj(output["TRN"]),
        xy_harmonics=sim_instance.sim_config.xy_harmonics,
    )

    return result.get_result_using_config(sim_instance.sim_config)
