import pytest
import tensorflow as tf
import numpy as np
from metabox import rcwa
from metabox.utils import Incidence, Feature

__author__ = "Luocheng Huang"
__copyright__ = "Luocheng Huang"
__license__ = "MIT"


def test_get_features():
    feature = Feature(0, 1, "feature")
    rect = rcwa.Rectangle(
        material=1.0,
        x_width=1,
        y_width=1,
        rotation_deg=0,
    )
    features = list(rect.get_features())
    assert len(features) == 0  # no Feature in Shape

    feature = Feature(0, 1, "a feature")
    rect = rcwa.Rectangle(
        material=1.0,
        x_width=feature,
        y_width=1,
        rotation_deg=0,
    )
    features = list(rect.get_features())
    assert len(features) == 1  # 1 Feature in rect


@pytest.mark.parametrize(
    "shapes,x_width,y_width,rotation_deg",
    [
        ([], 1.0, 1.0, 0.0),
        (
            [rcwa.Polygon(1.0, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])],
            1.0,
            1.0,
            0.0,
        ),
        (
            [rcwa.Polygon(1.0, [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)])],
            2.0,
            2.0,
            45.0,
        ),
    ],
)
def test_Rectangle(shapes, x_width, y_width, rotation_deg):
    rect = rcwa.Rectangle(
        material=1.0,
        x_width=x_width,
        y_width=y_width,
        rotation_deg=rotation_deg,
    )

    assert rect.x_width == x_width
    assert rect.y_width == y_width
    assert rect.rotation_deg == rotation_deg


@pytest.mark.parametrize(
    "radius,x_pos,y_pos",
    [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
    ],
)
def test_Circle(radius, x_pos, y_pos):
    circle = rcwa.Circle(material=1.0, radius=radius, x_pos=x_pos, y_pos=y_pos)

    assert circle.radius == radius
    assert circle.x_pos == x_pos
    assert circle.y_pos == y_pos


def test_PeriodicUnitCell():
    material = Feature(1, 2, "material")
    radius = Feature(0.5, 1.5, "radius")
    x_pos = Feature(-1, 1, "x_pos")
    y_pos = Feature(-1, 1, "y_pos")
    circle = rcwa.Circle(
        material=material, radius=radius, x_pos=x_pos, y_pos=y_pos
    )

    x_width = Feature(1, 2, "x_width")
    y_width = Feature(1, 2, "y_width")
    rotation_deg = Feature(0, 180, "rotation_deg")
    rectangle = rcwa.Rectangle(
        material=material,
        x_width=x_width,
        y_width=y_width,
        rotation_deg=rotation_deg,
    )

    thickness = Feature(0, 1, "thickness")
    layer = rcwa.Layer(
        material=material, thickness=thickness, shapes=(circle, rectangle)
    )

    periodicity = (Feature(1, 2, "x_period"), Feature(1, 2, "y_period"))
    refl_index = Feature(1, 2, "refl_index")
    tran_index = Feature(1, 2, "tran_index")
    unit_cell = rcwa.UnitCell(
        layers=[layer, layer],
        periodicity=periodicity,
        refl_index=refl_index,
        tran_index=tran_index,
    )

    assert unit_cell.layers == [layer, layer]
    assert unit_cell.periodicity == periodicity
    assert unit_cell.refl_index == refl_index
    assert unit_cell.tran_index == tran_index


def test_SimConfig():
    xy_harmonics = (1, 1)
    resolution = 100
    minibatch_size = 10
    sim_config = rcwa.SimConfig(
        xy_harmonics=xy_harmonics,
        resolution=resolution,
        minibatch_size=minibatch_size,
    )

    assert sim_config.xy_harmonics == xy_harmonics
    assert sim_config.resolution == resolution
    assert sim_config.minibatch_size == minibatch_size
