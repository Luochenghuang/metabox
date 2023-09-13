import pytest, copy
import tensorflow as tf
import numpy as np
from metabox import propagation


@pytest.fixture
def field():
    return propagation.Field2D(
        n_pixels=5,
        wavelength=np.linspace(460e-9, 660e-9, 3),
        theta=[0],
        phi=[0],
        period=1.0,
        upsampling=1,
        use_padding=False,
        use_antialiasing=True,
        tensor=tf.ones((3, 5, 5), dtype=tf.complex64),
    )


@pytest.fixture
def field_2():
    return propagation.Field2D(
        n_pixels=3,
        wavelength=np.linspace(460e-9, 660e-9, 3),
        theta=[0],
        phi=[0],
        period=1,
        upsampling=1,
        use_padding=False,
        use_antialiasing=True,
        tensor=tf.ones((3, 3, 3), dtype=tf.complex64),
    )


@pytest.fixture
def field_3():
    return propagation.Field2D(
        n_pixels=5,
        wavelength=np.linspace(460e-9, 660e-9, 3),
        theta=[0],
        phi=[0],
        period=0.5,
        upsampling=1,
        use_padding=False,
        use_antialiasing=True,
        tensor=tf.ones((3, 5, 5), dtype=tf.complex64),
    )


class TestField2D:
    def test_show_phase(self, field):
        field.show_phase()

    def test_show_intensity(self, field):
        field.show_intensity()

    def test_show_color_intensity(self, field):
        field.show_color_intensity()

    def test_modulated_by(self, field, field_2):
        out = field.modulated_by(field_2)
        assert out.tensor.shape == (3, 5, 5)

    def test_modulated_by_2(self, field, field_3):
        out = field.modulated_by(field_3)
        assert out.period == 0.5
        assert out.tensor.shape == (3, 5, 5)
