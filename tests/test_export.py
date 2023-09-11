import pytest
from metabox import export, rcwa, assembly, utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def test_circular():
    cell_period = 442e-9

    radius = utils.Feature(vmin=0, vmax=221e-9, name="radius")
    circle_1 = rcwa.Circle(2, radius=radius)
    patterned_layer = rcwa.Layer(1, thickness=632e-9, shapes=[circle_1])
    substrate = rcwa.Layer(1.5, thickness=632e-9)
    cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate],
        periodicity=(cell_period, cell_period),
    )
    protocell = rcwa.ProtoUnitCell(cell)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=5e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        proto_unit_cell=protocell,
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=False,  # set the structures as a variable to optimize
        use_circular_expansions=True,
    )

    export.generate_gds(metasurface, layer=0, export_name="test_circular")


def test_noncircular():
    cell_period = 442e-9

    radius = utils.Feature(vmin=0, vmax=221e-9, name="radius")
    circle_1 = rcwa.Circle(2, radius=radius)
    patterned_layer = rcwa.Layer(1, thickness=632e-9, shapes=[circle_1])
    substrate = rcwa.Layer(1.5, thickness=632e-9)
    cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate],
        periodicity=(cell_period, cell_period),
    )
    protocell = rcwa.ProtoUnitCell(cell)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=5e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        proto_unit_cell=protocell,
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=False,  # set the structures as a variable to optimize
        use_circular_expansions=False,
    )

    export.generate_gds(metasurface, layer=0, export_name="test_noncircular")


def test_4fold_symmetry():
    cell_period = 442e-9

    radius = utils.Feature(vmin=0, vmax=221e-9, name="radius")
    circle_1 = rcwa.Circle(2, radius=radius)
    patterned_layer = rcwa.Layer(
        1, thickness=632e-9, shapes=[circle_1], enforce_4fold_symmetry=True
    )
    substrate = rcwa.Layer(1.5, thickness=632e-9)
    cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate],
        periodicity=(cell_period, cell_period),
    )
    protocell = rcwa.ProtoUnitCell(cell)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=5e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        proto_unit_cell=protocell,
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=False,  # set the structures as a variable to optimize
        use_circular_expansions=False,
    )

    export.generate_gds(metasurface, layer=0, export_name="test_noncircular")
