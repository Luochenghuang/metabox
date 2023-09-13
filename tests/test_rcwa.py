import pytest
import numpy as np
from metabox import rcwa, utils, modeling

__author__ = "Luocheng Huang"
__copyright__ = "Luocheng Huang"
__license__ = "MIT"


def test_get_features():
    feature = rcwa.Feature(0, 1, "feature")
    rect = rcwa.Rectangle(
        material=1.0,
        x_width=1,
        y_width=1,
        rotation_deg=0,
    )
    features = list(rect.get_features())
    assert len(features) == 0  # no Feature in Shape

    feature = rcwa.Feature(0, 1, "a feature")
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
    material = rcwa.Feature(1, 2, "material")
    radius = rcwa.Feature(0.5, 1.5, "radius")
    x_pos = rcwa.Feature(-1, 1, "x_pos")
    y_pos = rcwa.Feature(-1, 1, "y_pos")
    circle = rcwa.Circle(
        material=material, radius=radius, x_pos=x_pos, y_pos=y_pos
    )

    x_width = rcwa.Feature(1, 2, "x_width")
    y_width = rcwa.Feature(1, 2, "y_width")
    rotation_deg = rcwa.Feature(0, 180, "rotation_deg")
    rectangle = rcwa.Rectangle(
        material=material,
        x_width=x_width,
        y_width=y_width,
        rotation_deg=rotation_deg,
    )

    thickness = rcwa.Feature(0, 1, "thickness")
    layer = rcwa.Layer(
        material=material, thickness=thickness, shapes=(circle, rectangle)
    )

    periodicity = (
        rcwa.Feature(1, 2, "x_period"),
        rcwa.Feature(1, 2, "y_period"),
    )
    refl_index = rcwa.Feature(1, 2, "refl_index")
    tran_index = rcwa.Feature(1, 2, "tran_index")
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


class TestParameterizable:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.feature1 = rcwa.Feature(0.0, 1.0, "feature1")
        self.feature2 = rcwa.Feature(0.0, 1.0, "feature2")
        self.parameterizable = rcwa.Polygon(
            1.0, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        )

    def test_initialize_values(self):
        with pytest.raises(ValueError):
            self.parameterizable.initialize_values(("invalid",))

        with pytest.raises(ValueError):
            self.parameterizable.initialize_values(
                ([self.feature1], [0.5, 0.6])
            )

        self.parameterizable.initialize_values(
            ([self.feature1, self.feature2], [0.5, 0.6])
        )
        assert self.feature1.initial_value == 0.5
        assert self.feature2.initial_value == 0.6

    def test_replace_feature_with_value(self):
        self.parameterizable.initialize_values(
            ([self.feature1, self.feature2], [0.5, 0.6])
        )
        self.parameterizable.replace_feature_with_value()


class TestShape:
    def test_shape_initialization(self):
        shape = rcwa.Shape(None)
        assert shape.material == None

    def test_polygon(self):
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        polygon = rcwa.Polygon(1.0, vertices)
        assert polygon.material == 1.0
        assert polygon.vertices == vertices

    def test_circle(self):
        circle = rcwa.Circle(1.0, 1.0, 0.0, 0.0)
        assert circle.material == 1.0
        assert circle.radius == 1.0
        assert circle.x_pos == 0.0
        assert circle.y_pos == 0.0

    def test_rectangle(self):
        rectangle = rcwa.Rectangle(1.0, 1.0, 1.0, 0.0)
        assert rectangle.material == 1.0
        assert rectangle.x_width == 1.0
        assert rectangle.y_width == 1.0
        assert rectangle.rotation_deg == 0.0


def test_duplicate_shape():
    rect = rcwa.Rectangle(
        material=1.0,
        x_width=1,
        y_width=1,
        rotation_deg=0,
    )
    rect2 = rcwa.duplicate_shape(rect, 5)[0]
    assert rect2.material == rect.material
