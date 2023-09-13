from metabox import utils
import tensorflow as tf


class TestFeature:
    def setup_method(self):
        self.feature = utils.Feature(0.0, 1.0, "test")

    def test_initialize_value(self):
        self.feature.initialize_value()
        assert self.feature.value is not None
        assert tf.is_tensor(self.feature.value)

    def test_set_variable(self):
        self.feature.initialize_value()
        self.feature.set_variable()
        assert isinstance(self.feature.value, tf.Variable)

    def test_set_value(self):
        self.feature.set_value(0.5)
        assert self.feature.value == 0.5
