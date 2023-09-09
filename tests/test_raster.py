import pytest
import numpy as np
import tensorflow as tf
from metabox.raster import Shape, Polygon, Circle, Rectangle, Canvas

__author__ = "Luocheng Huang"
__copyright__ = "Luocheng Huang"
__license__ = "MIT"


def test_shapes():
    # Test shapes with valid parameters
    s = Shape(1.0)
    assert s.value == 1.0
    assert not s.use_complex

    s = Shape(1 + 1j)
    assert s.value == 1 + 1j
    assert s.use_complex


def test_polygon():
    # Test polygon with valid parameters
    p = Polygon(1.0, [(0, 0), (1, 0), (0, 1)])
    assert p.value == 1.0
    assert not p.use_complex
    assert len(p.points) == 3

    # Test polygon with invalid parameters
    with pytest.raises(ValueError):
        p = Polygon(1.0, [(0, 0), (1, 0)])  # not enough points


def test_circle():
    # Test circle with valid parameters
    c = Circle(1.0, (0, 0), 1.0)
    assert c.value == 1.0
    assert not c.use_complex
    assert c.center == (0.0, 0.0)
    assert c.radius == 1.0


def test_rectangle():
    # Test rectangle with valid parameters
    r = Rectangle(1.0, (0, 0), 1.0, 1.0)
    assert r.value == 1.0
    assert not r.use_complex
    assert r.center == (0.0, 0.0)
    assert r.x_width == 1.0
    assert r.y_width == 1.0
    assert r.rotation_deg == 0.0  # default rotation


def test_canvas():
    # Test canvas with valid parameters
    c = Canvas(10.0, 10.0)
    assert c.x_width == 10.0
    assert c.y_width == 10.0
    assert c.spacing == 1.0  # default spacing
    assert c.background_value == 0.0  # default background value
    assert not c.enforce_4fold_symmetry  # default enforce_4fold_symmetry
    assert c.x_pixels == 10
    assert c.y_pixels == 10

    # Test rasterize with valid parameters
    p = Polygon(1.0, [(0, 0), (1, 0), (0, 1)])
    c.rasterize([p])

    # Test merge with valid parameters
    c2 = Canvas(10.0, 10.0)
    c.merge_with(c2)

    # Test draw
    c.draw()
