"""
Defines functions to expand a 1d field to a 2d field.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle

import numpy as np
import tensorflow as tf


def expand_to_2d(tensor: tf.Tensor, basis_dir="basis_data") -> tf.Tensor:
    """Function to expand a 1d field to a 2d field.

    Args:
        tensor (tf.tensor): the 1d field to expand
        basis_dir: the directory where the basis is saved.
            The default directory is "basis_data".

    Returns:
        tf.Tensor: the expanded field tensor.
    """

    # retrieve the basis
    radius_size = tensor.shape[-1]
    basis = load_basis(radius_size * 2, basis_dir)

    # Convert the RCWA output to a field
    radial_arr = tf.cast(tensor, tf.complex64)
    # radial_arr = tf.sparse.from_dense(radial_arr)

    # feed radial profile to the basis matrix to get the [batchsize, pixelsX, pixelsY] phase matrix
    circle = tf.sparse.sparse_dense_matmul(radial_arr, basis)
    circle = tf.math.conj(circle)  # this is a hack to get the correct phase
    return tf.reshape(circle, [-1, radius_size * 2, radius_size * 2])


def load_basis(n_pix, basis_dir=None) -> tf.Tensor:
    """retrieve the 1d to 2d basis from the basis_dir.

    If the basis is not found, it is created and saved to the basis_dir.

    Args:
        n_pix_radial (int): number of pixels per axis
        basis_dir (str): path to the directory where the basis is saved.
            if None, the basis is generated and not loaded or saved.

    Returns:
        tf.Tensor: the 1d to 2d basis.
    """

    # Initialize radius to circle basis
    basis_file_path = os.path.join(basis_dir, "r2c_basis_{}".format(n_pix // 2))
    if basis_dir is not None:
        if os.path.exists(basis_file_path):
            with open(basis_file_path, "rb") as picked_file:
                basis = pickle.load(picked_file)
            return tf.cast(basis, tf.complex64)
        else:
            print("Basis file not found. Creating basis.")

            basis_tensor = tf.cast(radius_to_circle_basis(n_pix // 2), tf.complex64)

            if not os.path.exists(basis_dir):
                os.makedirs(basis_dir)
            with open(basis_file_path, "wb") as output_file:
                pickle.dump(basis_tensor, output_file)
            return tf.cast(basis_tensor, tf.complex64)

    else:
        return tf.cast(radius_to_circle_basis(n_pix // 2), tf.complex64)


# initializes the radius to circle basis
def radius_to_circle_basis(radius_size) -> tf.Tensor:
    """Create a basis to map a 1d field to a 2d field.

    Args:
        radius_size (int): number of pixels in the radius

    Returns:
        tf.Tensor: the 1d to 2d basis
    """
    x = tf.linspace(-1, 1, radius_size * 2)
    y = tf.linspace(-1, 1, radius_size * 2)
    _, step = np.linspace(-1, 1, radius_size * 2, retstep=True)
    r = np.arange(0 + step, 1 + step, step)
    xx, yy = tf.meshgrid(x, y)
    rr = tf.sqrt(xx**2 + yy**2)
    output_list = []
    for idx, current_r in enumerate(r):
        if idx > 0:
            previous_r = r[idx - 1]
        else:
            previous_r = 0
        outer_circle = tf.cast(current_r >= rr, tf.int32)
        inner_circle = tf.cast(previous_r >= rr, tf.int32)
        ring = outer_circle - inner_circle
        ring = tf.squeeze(ring)
        ring = tf.sparse.from_dense([ring])
        # basis_tensor[:, :, idx] = ring
        output_list.append(ring)
    output_list = tf.sparse.concat(sp_inputs=output_list, axis=0)
    return tf.sparse.reshape(output_list, [radius_size, -1])
