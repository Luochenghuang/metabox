"""
This module defines classes and functions to simulate meta atoms.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import S4
import multiprocessing
import tqdm
import copy
import itertools
import tensorflow as tf
import numpy as np
from typing import List, Union
import dataclasses
import pandas
import pickle
import json
from utils import Feature

ParameterType = Union[Feature, float, tf.Tensor]


@dataclasses.dataclass
class Atom:
    """Defines a meta atom.

    An attribute of the atom can be a feature of a list of a combination of features
    and floats.

    Args:
        period: the period of the meta atom in meters.
        wavelength: the wavelength of the light in meters.
        polarization_sensitive: whether the atom is polarization sensitive.
            If True, both S and P polarizations will be used.
            If False, only S polarization will be used.
    """

    period: ParameterType
    wavelength: ParameterType
    polarization_sensitive: bool

    def __post_init__(self):
        self.feature_attrs = []
        self.unique_features = []
        for (key, value) in dataclasses.asdict(self).items():
            if isinstance(getattr(self, key), Feature):
                self.unique_features.append(getattr(self, key))
                self.feature_attrs.append(key)
        self.unique_features = list(set(self.unique_features))

    def to_dict(self):
        """Converts the atom to a dictionary.

        Returns:
            A dictionary representation of the atom.
        """
        return dataclasses.asdict(self)

    def get_features(self) -> List[Feature]:
        """Gets the features of the atom.

        Returns:
            A list of features.
        """
        return [getattr(self, attr) for attr in self.feature_attrs]

    def gen_atom_from_values(self, values) -> None:
        """Generates a new atom with the given values corresponding to the features.

        For example, if the atom has two features, the values must be a list of length 2.
        A new atom is then generated with the values of the features set to the values
        in the list. Each attribute of the original atom can be a feature of
        a list of features and floats.

        Args:
            values: a list of values to set the features to.
        """
        if len(values) != len(self.unique_features):
            raise ValueError(
                "The number of values must be equal to the number of features."
            )

        new_self = copy.deepcopy(self)
        # set the values of the features
        for feature, value in zip(new_self.unique_features, values):
            feature.value = value

        # replace the features with their values
        for (key, value) in dataclasses.asdict(new_self).items():
            this_attr = getattr(new_self, key)
            if isinstance(this_attr, Feature):
                setattr(new_self, key, this_attr.value)
            elif isinstance(this_attr, list):
                new_list = []
                for item in this_attr:
                    if isinstance(item, Feature):
                        new_list.append(item.value)
                    else:
                        new_list.append(item)
                setattr(new_self, key, new_list)

        return new_self


@dataclasses.dataclass
class Layer:
    """Defines a layer.

    Args:
        thickness: the thickness of the layer in meters.
    """

    thickness: Union[Feature, float]
    index: Union[Feature, float]


@dataclasses.dataclass
class UnitCell(Atom):
    in_transmission: bool
    layers: List[Layer]
    truncation_order: int = 100


@dataclasses.dataclass
class CrossPillar(Atom):
    """Defines a square pillar.

    A square pillar has a substrate and a pillar. The pillar has a square top cross-section.
    A buffer layer is placed between the pillar and the substrate who has the same refractive index as the pillar.
    The following diagram shows the side cross-section of the permittivity distribution.

    +---------------------------+
    |  pillar_background_index  |
    |                           |
    |                           |
    |      +-------------+      |
    |      |             |      |
    |      |pillar_index |      |
    |      |             |      |
    |------+             +------|
    |                           |
    |---------------------------|
    |                           |
    |      substrate_index      |
    |                           |
    |---------------------------|
    |                           |
    |substrate_background_index |
    |                           |
    +---------------------------+

    The following diagram shows the top view of the permittivity distribution,
        as well as the parameterization variables.

                        dd
                     ◄──────►

            ┌────────────────────────┐
            │                        │
         ▲  │        ┌──────┐        │
         │  │        │      │        │
         │  │        │      │        │
         │  │  ┌─────┘      └─────┐  │   ▲
    aa   │  │  │                  │  │   │ cc
         │  │  │                  │  │   │
         │  │  └─────┐      ┌─────┘  │   ▼
         │  │        │      │        │
         │  │        │      │        │
         ▼  │        └──────┘        │
            │                        │
            └────────────────────────┘

               ◄──────────────────►
                        bb

    Args:
        period: the period of the square pillar in meters.
        wavelength: the wavelength of the light in meters.
        pillar_index: the refractive index of the pillar.
        pillar_background_index: the refractive index of the background of the pillar.
        pillar_height: the height of the pillar in meters.
        aa: the length of the x span of the cross pillar in meters.
        bb: the length of the y span of the cross pillar in meters.
        cc: the length of the x width of the cross pillar in meters.
        dd: the length of the y width of the cross pillar in meters.
        buffer_thickness: the thickness of the buffer layer in meters.
        substrate_index: the refractive index of the substrate.
        substrate_background_index: the refractive index of the background of the substrate.
        use_transmission: whether to use the transmission or reflection mode.
        incident_angle: the incident angle of the light in radians.
        truncation_order: the truncation order of the Fourier expansion.
    """

    pillar_index: Union[Feature, float]
    pillar_background_index: Union[Feature, float]
    pillar_height: Union[Feature, float]
    aa: Union[Feature, float]
    bb: Union[Feature, float]
    cc: Union[Feature, float]
    dd: Union[Feature, float]
    buffer_thickness: Union[Feature, float]
    substrate_index: Union[Feature, float]
    substrate_background_index: Union[Feature, float]
    substrate_thickness: Union[Feature, float]
    use_transmission: bool
    incident_angle: Union[Feature, float] = 0.0
    truncation_order: int = 100


@dataclasses.dataclass
class SquarePillar(Atom):
    """Defines a square pillar.

    A square pillar has a substrate and a pillar. The pillar has a square top cross-section.
    A buffer layer is placed between the pillar and the substrate who has the same refractive index as the pillar.
    The following diagram shows the side cross-section of the permittivity distribution.

    +---------------------------+
    |  pillar_background_index  |
    |                           |
    |                           |
    |      +-------------+      |
    |      |             |      |
    |      |pillar_index |      |
    |      |             |      |
    |------+             +------|
    |                           |
    |---------------------------|
    |                           |
    |      substrate_index      |
    |                           |
    |---------------------------|
    |                           |
    |substrate_background_index |
    |                           |
    +---------------------------+

    Args:
        period: the period of the square pillar in meters.
        wavelength: the wavelength of the light in meters.
        pillar_index: the refractive index of the pillar.
        pillar_background_index: the refractive index of the background of the pillar.
        pillar_height: the height of the pillar in meters.
        pillar_width: the width of the pillar in meters.
        buffer_thickness: the thickness of the buffer layer in meters.
        substrate_index: the refractive index of the substrate.
        substrate_background_index: the refractive index of the background of the substrate.
        use_transmission: whether to use the transmission or reflection mode.
        incident_angle: the incident angle of the light in radians.
        truncation_order: the truncation order of the Fourier expansion.
    """

    pillar_index: Union[Feature, float]
    pillar_background_index: Union[Feature, float]
    pillar_height: Union[Feature, float]
    pillar_width: Union[Feature, float]
    buffer_thickness: Union[Feature, float]
    substrate_index: Union[Feature, float]
    substrate_background_index: Union[Feature, float]
    substrate_thickness: Union[Feature, float]
    use_transmission: bool
    incident_angle: Union[Feature, float] = 0.0
    truncation_order: int = 100


@dataclasses.dataclass
class AtomLibrary:
    """Defines a simulated library of meta atoms.

    Args:
        dataframe: the dataframe that stores the meta atom data.
        proto_atom: the prototype meta atom with built-in features.
        description (optional): the description of the library.
    """

    dataframe: pandas.DataFrame
    proto_atom: Atom
    description: str = "No description provided."

    def get_training_data(self):
        """Generates the training data for the meta atom library.

        The training data is a tuple of the input features and the output coefficients.
        The input features are the feature values of the meta atoms.
        The output coefficients are the complex transmission coefficients of the meta atoms.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the training data.
        """
        feature_keys = [
            feature.name for feature in self.proto_atom.unique_features
        ]
        input_features = self.dataframe[feature_keys].to_numpy()
        output = self.dataframe["coefficient"].to_numpy()
        return (input_features, output)

    def save(
        self,
        name: str,
        path: str = "./saved_atom_libraries",
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
        pickle.dump(self, filehandler_pkl)
        json_dict = self.proto_atom.to_dict()
        for item in json_dict:
            if np.iscomplex(json_dict[item]):
                json_dict[item] = str(json_dict[item])
        json_d = json.dumps(json_dict, indent=4)
        with open(full_path + ".json", "w") as outfile:
            outfile.write(json_d)

        print("Saved the atom library to " + full_path)


def simulate_cross_pillar(
    atom: CrossPillar,
) -> float:
    """Function to simulate the propagation of a meta atom.

    Args:
        atom: the meta atom to simulate.

    Returns:
        The complex transmission coefficient of the structure.
    """

    S = S4.New(
        Lattice=((atom.period, 0), (0, atom.period)),
        NumBasis=atom.truncation_order,
    )
    # set materials
    S.SetMaterial(Name="pillar", Epsilon=atom.pillar_index**2)
    S.SetMaterial(Name="pillar_bg", Epsilon=atom.pillar_background_index**2)
    S.SetMaterial(Name="substrate", Epsilon=atom.substrate_index**2)
    S.SetMaterial(
        Name="substrate_bg", Epsilon=atom.substrate_background_index**2
    )

    # set layers
    S.AddLayer(Name="pillar_bg", Thickness=1.0, Material="pillar_bg")
    S.AddLayer(
        Name="meta_atom", Thickness=atom.pillar_height, Material="pillar_bg"
    )

    vertices = (
        (atom.bb / 2.0, atom.cc / 2.0),
        (atom.dd / 2.0, atom.cc / 2.0),
        (atom.dd / 2.0, atom.aa / 2.0),
        (-atom.dd / 2.0, atom.aa / 2.0),
        (-atom.dd / 2.0, atom.cc / 2.0),
        (-atom.bb / 2.0, atom.cc / 2.0),
        (-atom.bb / 2.0, -atom.cc / 2.0),
        (-atom.dd / 2.0, -atom.cc / 2.0),
        (-atom.dd / 2.0, -atom.aa / 2.0),
        (atom.dd / 2.0, -atom.aa / 2.0),
        (atom.dd / 2.0, -atom.cc / 2.0),
        (atom.bb / 2.0, -atom.cc / 2.0),
    )

    S.SetRegionPolygon(
        Layer="meta_atom",
        Material="pillar",
        Center=(0, 0),
        Angle=0,  # in degrees
        Vertices=vertices,
    )
    S.AddLayer(
        Name="buffer", Thickness=atom.buffer_thickness, Material="pillar"
    )
    S.AddLayer(
        Name="substrate",
        Thickness=atom.substrate_thickness,
        Material="substrate",
    )
    S.AddLayer(
        Name="substrate_bg",
        Thickness=1.0,
        Material="substrate_bg",
    )

    # set excitation, assume p polarized. The result is the same for s polarized
    # since the structure is symmetric.
    p_amp, s_amp = 0, 1
    S.SetExcitationPlanewave(
        IncidenceAngles=(
            atom.incident_angle,  # polar angle in [0,180)
            0,  # azimuthal angle in [0,360)
        ),
        sAmplitude=s_amp,
        pAmplitude=p_amp,
        Order=0,
    )

    S.SetFrequency(1 / atom.wavelength)
    if atom.use_transmission:
        (forw_field, back_field) = S.GetAmplitudes(
            Layer="substrate_bg", zOffset=0
        )
        return forw_field[0]
    else:
        (forw_field, back_field) = S.GetAmplitudes(
            Layer="pillar_bg", zOffset=0
        )
        return back_field[0]


def load_atom_library(
    name: str,
    path: str = "./saved_atom_libraries",
) -> AtomLibrary:
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


def simulate_square_pillar(
    atom: SquarePillar,
) -> float:
    """Function to simulate the propagation of a meta atom.

    Args:
        atom: the meta atom to simulate.

    Returns:
        The complex transmission coefficient of the structure.
    """

    S = S4.New(
        Lattice=((atom.period, 0), (0, atom.period)),
        NumBasis=atom.truncation_order,
    )
    # set materials
    S.SetMaterial(Name="pillar", Epsilon=atom.pillar_index**2)
    S.SetMaterial(Name="pillar_bg", Epsilon=atom.pillar_background_index**2)
    S.SetMaterial(Name="substrate", Epsilon=atom.substrate_index**2)
    S.SetMaterial(
        Name="substrate_bg", Epsilon=atom.substrate_background_index**2
    )

    # set layers
    S.AddLayer(Name="pillar_bg", Thickness=1.0, Material="pillar_bg")
    S.AddLayer(
        Name="meta_atom", Thickness=atom.pillar_height, Material="pillar_bg"
    )
    S.SetRegionRectangle(
        Layer="meta_atom",
        Material="pillar",
        Center=(0, 0),
        Angle=0,  # in degrees
        Halfwidths=(atom.pillar_width / 2, atom.pillar_width / 2),
    )
    S.AddLayer(
        Name="buffer", Thickness=atom.buffer_thickness, Material="pillar"
    )
    S.AddLayer(
        Name="substrate",
        Thickness=atom.substrate_thickness,
        Material="substrate",
    )
    S.AddLayer(
        Name="substrate_bg",
        Thickness=1.0,
        Material="substrate_bg",
    )

    # set excitation, assume p polarized. The result is the same for s polarized
    # since the structure is symmetric.
    p_amp, s_amp = 0, 1
    S.SetExcitationPlanewave(
        IncidenceAngles=(
            atom.incident_angle,  # polar angle in [0,180)
            0,  # azimuthal angle in [0,360)
        ),
        sAmplitude=s_amp,
        pAmplitude=p_amp,
        Order=0,
    )

    S.SetFrequency(1 / atom.wavelength)
    if atom.use_transmission:
        (forw_field, back_field) = S.GetAmplitudes(
            Layer="substrate_bg", zOffset=0
        )
        return forw_field[0]
    else:
        (forw_field, back_field) = S.GetAmplitudes(
            Layer="pillar_bg", zOffset=0
        )
        return back_field[0]


def sample_feature(feature: Feature) -> np.ndarray:
    """Function to sample a feature.

    Args:
        feature: the feature to sample.
        sampling: the number of samples.

    Returns:
        A list of sampled values.
    """

    if feature is Feature:
        raise ValueError("The feature is not specified.")

    return np.linspace(feature.vmin, feature.vmax, feature.sampling)


def generate_square_pillar_library(
    proto_atom: SquarePillar,
) -> AtomLibrary:
    """Function to generate the library of a square pillar.

    Args:
        atom: the meta atom to simulate.
        feature_sampling: the number of samples for each feature.

    Returns:
        AtomLibrary: the simulated library of the meta atom.
    """
    print(
        "`generate_square_pillar_library` is deprecated. Use `generate_simulation_library` instead."
    )
    return generate_simulation_library(proto_atom)


def generate_simulation_library(
    proto_atom: Atom,
) -> AtomLibrary:
    """Function to generate the simulation library for meta-atoms.

    Args:
        atom: the meta atom to simulate.
        feature_sampling: the number of samples for each feature.

    Returns:
        AtomLibrary: the simulated library of the meta atom.
    """

    if len(proto_atom.feature_attrs) == 0:
        raise ValueError("The meta atom does not have any features.")

    # sample features
    feature_values_list = []
    for feature in proto_atom.unique_features:
        values = sample_feature(feature)
        feature_values_list.append(values)

    # generate all atom permutations
    value_permutations = list(itertools.product(*feature_values_list))
    all_atoms = []
    for value_permutation in value_permutations:
        new_atom = proto_atom.gen_atom_from_values(value_permutation)
        all_atoms.append(new_atom)

    if isinstance(proto_atom, SquarePillar):
        simulation_function = simulate_square_pillar
    elif isinstance(proto_atom, CrossPillar):
        simulation_function = simulate_cross_pillar
    else:
        raise ValueError("The meta atom is not supported.")

    # start simnulation
    p = multiprocessing.Pool()
    field_rtn = list(
        tqdm.tqdm(
            p.imap(simulation_function, all_atoms),
            total=len(all_atoms),
        )
    )
    p.close()  # prevent memory leakage
    p.join()  # synchronization point

    # create pandas dataframe and save
    keys = [feature.name for feature in proto_atom.unique_features]
    dataframe = pandas.DataFrame(list(value_permutations), columns=keys)
    dataframe["coefficient"] = np.array(field_rtn)

    return AtomLibrary(
        dataframe=dataframe,
        proto_atom=proto_atom,
    )
