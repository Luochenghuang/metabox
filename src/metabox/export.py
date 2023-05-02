"""
This module contains functions to generate the GDSII and other visualization files.
"""

import os
import importlib
from typing import Union
import numpy as np
import expansion
import assembly
import atom
import tensorflow as tf
import gdspy
import rcwa


def gen_unit_cell_square(this_atom: atom.SquarePillar):
    """Generates a unit cell for a square-shaped meta-atom.

    Args:
        this_atom: the meta-atom.

    Returns:
        gdspy.Rectangle: the unit cell.
    """
    if not isinstance(this_atom, atom.SquarePillar):
        raise TypeError(
            "The atom type '{}' is not supported.".format(type(this_atom))
        )
    a = this_atom.pillar_width * 1e6
    return gdspy.Rectangle((-a / 2, -a / 2), (a / 2, a / 2))


def gen_unit_cell_cross(this_atom: atom.CrossPillar):
    """Generates a unit cell for a cross-shaped meta-atom.

    Args:
        this_atom: the meta-atom.

    Returns:
        gdspy.Polygon: the unit cell.
    """

    aa = this_atom.aa * 1e6
    bb = this_atom.bb * 1e6
    cc = this_atom.cc * 1e6
    dd = this_atom.dd * 1e6

    # units in gdspy are in microns

    vertices = (
        (bb / 2.0, cc / 2.0),
        (dd / 2.0, cc / 2.0),
        (dd / 2.0, aa / 2.0),
        (-dd / 2.0, aa / 2.0),
        (-dd / 2.0, cc / 2.0),
        (-bb / 2.0, cc / 2.0),
        (-bb / 2.0, -cc / 2.0),
        (-dd / 2.0, -cc / 2.0),
        (-dd / 2.0, -aa / 2.0),
        (dd / 2.0, -aa / 2.0),
        (dd / 2.0, -cc / 2.0),
        (bb / 2.0, -cc / 2.0),
    )
    return gdspy.Polygon(vertices)


def gen_unit_cell(the_atom):
    """Generates unit cell from any meta-atom."""
    if isinstance(the_atom, atom.SquarePillar):
        return gen_unit_cell_square(the_atom)
    elif isinstance(the_atom, atom.CrossPillar):
        return gen_unit_cell_cross(the_atom)
    else:
        raise TypeError(
            "The atom type '{}' is not supported.".format(type(the_atom))
        )


def unit_cell_to_gds_shape(cell: rcwa.UnitCell, layer: int = 0):
    """Converts a unit cell to a GDSII shape.

    Args:
        cell (rcwa.UnitCell): the unit cell.

    Returns:
        gdspy.PolygonSet: the GDSII shape.
    """
    gds_shapes = []
    for shape in cell.layers[layer].shapes:
        vertices = []
        for vertex in shape.get_vertices():
            vertices.append((vertex[0] * 1e6, vertex[1] * 1e6))
        gds_shape = gdspy.PolygonSet([vertices])
        if cell.layers[layer].enforce_4fold_symmetry:
            gds_shape = gds_shape_force_4fold_symmetry(gds_shape)
        gds_shapes.append(gds_shape)

    shape = gds_shapes[0]
    for next_shape in gds_shapes[1:]:
        shape = gdspy.boolean(shape, next_shape, "or")
    return shape


def gds_shape_force_4fold_symmetry(
    shape: gdspy.PolygonSet,
) -> gdspy.PolygonSet:
    """Generates a 4-fold symmetric shape from a polygon"""
    new_shape = gdspy.copy(shape)
    new_shape = new_shape.rotate(np.pi / 2)
    shape = gdspy.boolean(shape, new_shape, "or")
    if shape is None:
        return None

    new_shape = gdspy.copy(shape)
    new_shape = new_shape.rotate(np.pi)
    shape = gdspy.boolean(shape, new_shape, "or")
    if shape is None:
        return None

    new_shape = gdspy.copy(shape)
    new_shape = new_shape.mirror((0, 1))
    return gdspy.boolean(shape, new_shape, "or")


def get_loc_along_circle(
    radius_in_meters, radius_size, query_radius, period, r2c_basis
):
    """
    inputs radius in micron, radius size, and the index of which radius you want to draw
    a circle using, outputs the x and y locations of the elements
    """
    radius_selection = np.zeros(radius_size)
    radius_selection[query_radius] = 1
    tf_radius = tf.reshape(radius_selection, [-1, radius_size])
    circle = tf.sparse.sparse_dense_matmul(tf_radius, r2c_basis)
    circle = tf.reshape(circle, [radius_size * 2, radius_size * 2])
    location_index_arr = np.argwhere(circle == 1)
    hp = period / 2
    radius_in_micron = radius_in_meters * 1e6
    x_coor = np.linspace(
        -radius_in_micron + hp, radius_in_micron - hp, radius_size * 2
    )
    y_coor = np.linspace(
        -radius_in_micron + hp, radius_in_micron - hp, radius_size * 2
    )
    x_locations = x_coor.take(location_index_arr[:, 0])
    y_locations = y_coor.take(location_index_arr[:, 1])
    return np.stack([x_locations, y_locations])


def generate_noncircular_gds(
    metasurface: assembly.Metasurface,
    export_name: str,
    export_directory: Union[None, str] = None,
    inverted: bool = False,
) -> None:
    """Function to generate the GDSII file for the non-circular metasurface.

    Args:
        metasurface (assembly.Metasurface): the metasurface.
        export_directory (str): the directory to export the GDSII file to.
        inverted (bool, optional): whether to invert the metasurface. Defaults to False.
    """
    if metasurface.use_circular_expansions:
        raise NotImplementedError(
            "generate_noncircular_gds is only implemented for non-circular expansions for now."
        )

    if export_directory is None:
        export_directory = "./gds_export/"

    importlib.reload(gdspy)

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()

    # the entire metasurface
    device = lib.new_cell("metasurface")

    periodicity = metasurface.metamodel.proto_atom.period * 1e6
    n_pixels_radial = metasurface.n_pixels_radial
    radius_in_meters = metasurface.diameter / 2.0
    radius_in_microns = radius_in_meters * 1e6
    dummy_incidence = assembly.Incidence(wavelength=1.0, theta=[0], phi=[0])
    atom_array = metasurface.atom_2d.get_atom_array(dummy_incidence)[0]
    x_locations = y_locations = np.linspace(
        -radius_in_microns, radius_in_microns, 2 * n_pixels_radial
    )
    all_locations = np.stack(np.meshgrid(x_locations, y_locations), axis=-1)
    all_locations = np.reshape(all_locations, (-1, 2))
    for radial_ix, (this_atom, locations) in enumerate(
        zip(atom_array, all_locations)
    ):
        scatterer = gen_unit_cell(this_atom)
        if inverted:
            box_hw = periodicity / 2.0
            box = gdspy.Rectangle([-box_hw, -box_hw], [box_hw, box_hw])
            scatterer = gdspy.boolean(box, scatterer, "not")
        cell = lib.new_cell(str(radial_ix))
        cell.add(scatterer)
        loc_x, loc_y = locations
        ref = gdspy.CellReference(cell, (loc_x, loc_y))
        device.add([ref])

    # save the GDSII file
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    lib.write_gds("{}/{}.gds".format(export_directory, export_name))


def generate_gds(
    metasurface: assembly.Metasurface,
    export_name: str,
    export_directory: Union[None, str] = None,
    inverted: bool = False,
) -> None:
    """Function to generate the GDSII file for the metasurface.

    Args:
        metasurface (assembly.Metasurface): the metasurface.
        export_directory (str): the directory to export the GDSII file to.
        inverted (bool, optional): whether to invert the metasurface. Defaults to False.
    """
    if not metasurface.use_circular_expansions:
        generate_noncircular_gds(
            metasurface=metasurface,
            export_name=export_name,
            export_directory=export_directory,
            inverted=inverted,
        )
        return

    if export_directory is None:
        export_directory = "./gds_export/"

    importlib.reload(gdspy)

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()

    # the entire metasurface
    device = lib.new_cell("metasurface")

    periodicity = metasurface.metamodel.proto_atom.period * 1e6
    n_pixels_radial = metasurface.n_pixels_radial
    radius_in_meters = metasurface.diameter / 2.0
    r2c_basis = expansion.radius_to_circle_basis(n_pixels_radial)
    r2c_basis = tf.cast(r2c_basis, tf.float64)
    dummy_incidence = assembly.Incidence(wavelength=1.0, theta=[0], phi=[0])
    atom_array = metasurface.atom_1d.get_atom_array(dummy_incidence)[0]
    for radial_ix, this_atom in enumerate(atom_array):
        scatterer = gen_unit_cell(this_atom)
        if inverted:
            box_hw = periodicity / 2.0
            box = gdspy.Rectangle([-box_hw, -box_hw], [box_hw, box_hw])
            scatterer = gdspy.boolean(box, scatterer, "not")
        cell = lib.new_cell(str(radial_ix))
        cell.add(scatterer)
        locations_along_circle = get_loc_along_circle(
            radius_in_meters,
            n_pixels_radial,
            radial_ix,
            periodicity,
            r2c_basis,
        )
        for index in range(locations_along_circle.shape[1]):
            loc_x, loc_y = locations_along_circle[:, index]
            ref = gdspy.CellReference(cell, (loc_x, loc_y))
            device.add([ref])

    # save the GDSII file
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    lib.write_gds("{}/{}.gds".format(export_directory, export_name))


def generate_noncircular_gds_new():
    raise NotImplementedError


def generate_gds_new(
    metasurface: assembly.Metasurface,
    layer: int,
    export_name: str,
    export_directory: Union[None, str] = None,
    inverted: bool = False,
) -> None:
    """Function to generate the GDSII file for the metasurface.

    Args:
        metasurface (assembly.Metasurface): the metasurface.
        layer: the layer in UnitCell to use for the shapes.
        export_directory (str): the directory to export the GDSII file to.
        inverted (bool, optional): whether to invert the metasurface. Defaults to False.
    """
    if not metasurface.use_circular_expansions:
        generate_noncircular_gds_new(
            metasurface=metasurface,
            export_name=export_name,
            export_directory=export_directory,
            inverted=inverted,
        )
        return

    if export_directory is None:
        export_directory = "./gds_export/"

    importlib.reload(gdspy)

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()

    # the entire metasurface
    device = lib.new_cell("metasurface")

    periodicity = metasurface.atom_1d.period * 1e6
    n_pixels_radial = metasurface.n_pixels_radial
    radius_in_meters = metasurface.diameter / 2.0
    r2c_basis = expansion.radius_to_circle_basis(n_pixels_radial)
    r2c_basis = tf.cast(r2c_basis, tf.float64)
    dummy_incidence = assembly.Incidence(wavelength=1.0, theta=[0], phi=[0])
    cell_array = metasurface.atom_1d.proto_unit_cell.generate_cells_from_parameter_tensor(
        metasurface.atom_1d.tensor
    )
    for radial_ix, this_cell in enumerate(cell_array):
        polygon = unit_cell_to_gds_shape(this_cell)
        if inverted:
            box_hw = periodicity / 2.0
            box = gdspy.Rectangle([-box_hw, -box_hw], [box_hw, box_hw])
            polygon = gdspy.boolean(box, polygon, "not")
        cell = lib.new_cell(str(radial_ix))
        cell.add(polygon)
        locations_along_circle = get_loc_along_circle(
            radius_in_meters,
            n_pixels_radial,
            radial_ix,
            periodicity,
            r2c_basis,
        )
        for index in range(locations_along_circle.shape[1]):
            loc_x, loc_y = locations_along_circle[:, index]
            ref = gdspy.CellReference(cell, (loc_x, loc_y))
            device.add([ref])

    # save the GDSII file
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    lib.write_gds("{}/{}.gds".format(export_directory, export_name))
