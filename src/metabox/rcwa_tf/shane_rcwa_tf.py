# Written by Shane Colburn (Email:
# Original work can be found here: https://doi.org/10.1038/s42005-021-00568-6

# BSD 3-Clause License
# Copyright (c) 2020, Shane Colburn, University of Washington
# All rights reserved.
# lburn (Email: scolbur2@uw.edu)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Tuple

import numpy as np
import tensorflow as tf

from metabox.utils import ParameterType


def expand_and_tile_np(array, batchSize, pixelsX, pixelsY):
    """
    Expands and tile a numpy array for a given batchSize and number of pixels.
    Args:
        array: A `np.ndarray` of shape `(Nx, Ny)`.
    Returns:
        A `np.ndarray` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `array` tiled over the new dimensions.
    """
    array = array[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.tile(array, reps=(batchSize, pixelsX, pixelsY, 1, 1))


def expand_and_tile_tf(tensor, batchSize, pixelsX, pixelsY):
    """
    Expands and tile a `tf.Tensor` for a given batchSize and number of pixels.
    Args:
        tensor: A `tf.Tensor` of shape `(Nx, Ny)`.
    Returns:
        A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `tensor` tiled over the new dimensions.
    """
    tensor = tensor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    return tf.tile(tensor, multiples=(batchSize, pixelsX, pixelsY, 1, 1))


@tf.custom_gradient
def eig_general(A, eps=1e-6):
    """
    Computes the eigendecomposition of a batch of matrices, the same as
    `tf.eig()` but assumes the input shape also has extra dimensions for pixels
    and layers. This function also provides the reverse mode gradient of the
    eigendecomposition as derived in 10.1109/ICASSP.2017.7952140. This applies
    for general, complex matrices that do not have to be self adjoint. This
    result gives the exact reverse mode gradient for nondegenerate eigenvalue
    problems. To extend to the case of degenerate eigenvalues common in RCWA, we
    approximate the gradient by a Lorentzian broadening technique that
    introduces a small error but stabilizes the calculation. This is based on
    10.1103/PhysRevX.9.031041.
    Args:
        A: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, n_layersers, Nx,
        Ny)` and dtype `tf.complex64` where the last two dimensions define
        matrices for which we will calculate the eigendecomposition of their
        reverse mode gradients.

        eps: A `float` defining a regularization parameter used in the
        denominator of the Lorentzian broadening calculation to enable reverse
        mode gradients for degenerate eigenvalues.

    Returns:
        A `Tuple(List[tf.Tensor, tf.Tensor], tf.Tensor)`, where the `List`
        specifies the eigendecomposition as computed by `tf.eig()` and the
        second element of the `Tuple` gives the reverse mode gradient of the
        eigendecompostion of the input argument `A`.
    """

    # Perform the eigendecomposition.
    eigenvalues, eigenvectors = tf.eig(A)

    # Referse mode gradient calculation.
    def grad(grad_D, grad_U):
        # Use the pre-computed eigendecomposition.
        nonlocal eigenvalues, eigenvectors
        D = eigenvalues
        U = eigenvectors

        # Convert eigenvalues gradient to a diagonal matrix.
        grad_D = tf.linalg.diag(grad_D)

        # Extract the tensor dimensions for later use.
        batchSize, pixelsX, pixelsY, n_layers, dim, _ = A.shape

        # Calculate intermediate matrices.
        I = tf.eye(num_rows=dim, dtype=tf.complex64)
        D = tf.reshape(D, shape=(batchSize, pixelsX, pixelsY, n_layers, dim, 1))
        shape_di = (batchSize, pixelsX, pixelsY, n_layers, dim, 1)
        shape_dj = (batchSize, pixelsX, pixelsY, n_layers, 1, dim)
        E = tf.ones(shape=shape_di, dtype=tf.complex64) * tf.linalg.adjoint(D)
        E = E - D * tf.ones(shape=shape_dj, dtype=tf.complex64)
        E = tf.linalg.adjoint(D) - D

        # Lorentzian broadening.
        F = E / (E**2 + eps)
        F = F - I * F

        # Compute the reverse mode gradient of the eigendecomposition of A.
        grad_A = tf.math.conj(F) * tf.linalg.matmul(tf.linalg.adjoint(U), grad_U)
        grad_A = grad_D + grad_A
        grad_A = tf.linalg.matmul(grad_A, tf.linalg.adjoint(U))
        grad_A = tf.linalg.matmul(tf.linalg.inv(tf.linalg.adjoint(U)), grad_A)
        return grad_A

    return [eigenvalues, eigenvectors], grad


def convmat(A, P, Q):
    """
    This function computes a convolution matrix for a real space matrix `A` that
    represents either a relative permittivity or permeability distribution for a
    set of pixels, layers, and batch.
    Args:
        A: A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX,
        pixelsY, n_layersers, Nx, Ny)` specifying real space values on a Cartesian
        grid.

        P: A positive and odd `int` specifying the number of spatial harmonics
        along `T1`.

        Q: A positive and odd `int` specifying the number of spatial harmonics
        along `T2`.
    Returns:
        A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX,
        pixelsY, n_layersers, P * Q, P * Q)` representing a stack of convolution
        matrices based on `A`.
    """

    # Determine the shape of A.
    batchSize, pixelsX, pixelsY, n_layersers, Nx, Ny = A.shape

    # Compute indices of spatial harmonics.
    NH = P * Q  # total number of harmonics.
    p_max = np.floor(P / 2.0)
    q_max = np.floor(P / 2.0)

    # Indices along T1 and T2.
    p = np.linspace(-p_max, p_max, P)
    q = np.linspace(-q_max, q_max, Q)

    # Compute array indices of the center harmonic.
    p0 = int(np.floor(Nx / 2))
    q0 = int(np.floor(Ny / 2))

    # Fourier transform the real space distributions.
    A = tf.signal.fftshift(tf.signal.fft2d(A), axes=(4, 5)) / (Nx * Ny)

    # Build the matrix.
    firstCoeff = True
    for qrow in range(Q):
        for prow in range(P):
            for qcol in range(Q):
                for pcol in range(P):
                    pfft = int(p[prow] - p[pcol])
                    qfft = int(q[qrow] - q[qcol])

                    # Sequentially concatenate Fourier coefficients.
                    value = A[:, :, :, :, p0 + pfft, q0 + qfft]
                    value = value[:, :, :, :, tf.newaxis, tf.newaxis]
                    if firstCoeff:
                        firstCoeff = False
                        C = value
                    else:
                        C = tf.concat([C, value], axis=5)

    # Reshape the coefficients tensor into a stack of convolution matrices.
    convMatrixShape = (batchSize, pixelsX, pixelsY, n_layersers, P * Q, P * Q)
    matrixStack = tf.reshape(C, shape=convMatrixShape)

    return matrixStack


def redheffer_star_product(SA, SB):
    """
    This function computes the redheffer star product of two block matrices,
    which is the result of combining the S-parameter of two systems.
    Args:
        SA: A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of a system. `SA` needs to have the
        keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a `tf.Tensor`
        of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where NH is the
        total number of spatial harmonics.

        SB: A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of a second system. `SB` needs to have
        the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a
        `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where
        NH is the total number of spatial harmonics.
    Returns:
        A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of the combined system. `SA` needs
        to have the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to
        a `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH),
        where NH is the total number of spatial harmonics.
    """
    # Define the identity matrix.
    batchSize, pixelsX, pixelsY, dim, _ = SA["S11"].shape
    I = tf.eye(num_rows=dim, dtype=tf.complex64)
    I = I[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    I = tf.tile(I, multiples=(batchSize, pixelsX, pixelsY, 1, 1))

    # Calculate S11.
    S11 = tf.linalg.inv(I - tf.linalg.matmul(SB["S11"], SA["S22"]))
    S11 = tf.linalg.matmul(S11, SB["S11"])
    S11 = tf.linalg.matmul(SA["S12"], S11)
    S11 = SA["S11"] + tf.linalg.matmul(S11, SA["S21"])

    # Calculate S12.
    S12 = tf.linalg.inv(I - tf.linalg.matmul(SB["S11"], SA["S22"]))
    S12 = tf.linalg.matmul(S12, SB["S12"])
    S12 = tf.linalg.matmul(SA["S12"], S12)

    # Calculate S21.
    S21 = tf.linalg.inv(I - tf.linalg.matmul(SA["S22"], SB["S11"]))
    S21 = tf.linalg.matmul(S21, SA["S21"])
    S21 = tf.linalg.matmul(SB["S21"], S21)

    # Calculate S22.
    S22 = tf.linalg.inv(I - tf.linalg.matmul(SA["S22"], SB["S11"]))
    S22 = tf.linalg.matmul(S22, SA["S22"])
    S22 = tf.linalg.matmul(SB["S21"], S22)
    S22 = SB["S22"] + tf.linalg.matmul(S22, SB["S12"])

    # Store S parameters in an output dictionary.
    S = dict({})
    S["S11"] = S11
    S["S12"] = S12
    S["S21"] = S21
    S["S22"] = S22

    return S


def simulate_rcwa(
    incidence_dict: dict,
    PQ: Tuple[int, int],
    n_cells: int,
    n_layers: int,
    layer_thicknesses: tf.Tensor,
    L_xy: Tuple[ParameterType, ParameterType],
    er1: ParameterType,
    er2: ParameterType,
    ER_t: tf.Tensor,
    UR_t: tf.Tensor,
    refl_n: ParameterType,
):
    """Simulates the periodic unit cell using RCWA.

    Args:
        incidence_dict: A `dict` of `tf.Tensor` values specifying the
            incidence parameters. `incidence_dict` needs to have the keys
            ('wavelength', 'theta', 'phi', 'polarization')
        PQ: A `tuple` of `int` values specifying the number of spatial
            harmonics in the x and y directions.
        n_cells: An `int` specifying the number of unit cells in the x
            direction.
        n_layers: An `int` specifying the number of layers in the unit cell.
        layer_thicknesses: A `tf.Tensor` of shape `(n_layers, )` specifying
            the thickness of each layer in the unit cell.
        L_xy: A `tuple` of `ParameterType` values specifying the size of the
            unit cell in the x and y directions.
        er1: A `ParameterType` specifying the relative permittivity of the
            transmission region in the unit cell.
        er2: A `ParameterType` specifying the relative permittivity of the
            reflection region in the unit cell.
        ER_t: A `tf.Tensor` specifying the relative permittivity of each
            layer in the unit cell.
        UR_t: A `tf.Tensor` specifying the relative permeability of each
            layer in the unit cell.
        refl_n: A `ParameterType` specifying the refractive index of the
            reflection region in the unit cell.
    Returns:
        The diffraction coefficients of the unit cell.
    """

    ### Step 1: Precompute some constants ###
    batch_size = len(incidence_dict["wavelength"])
    # Number of pixels in the x and y directions, set to 1 for now.
    n_cells_x = n_cells
    n_cells_y = 1

    # Batch parameters (wavelength, incidence angle, and polarization).
    lam0 = incidence_dict["wavelength"]
    lam0 = lam0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    lam0 = tf.tile(lam0, multiples=(1, n_cells_x, n_cells_y, 1, 1, 1))

    theta = incidence_dict["theta"]
    theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    theta = tf.tile(theta, multiples=(1, n_cells_x, n_cells_y, 1, 1, 1))

    phi = incidence_dict["phi"]
    phi = phi[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    phi = tf.tile(phi, multiples=(1, n_cells_x, n_cells_y, 1, 1, 1))

    pte = incidence_dict["pte"]
    pte = pte[:, tf.newaxis, tf.newaxis, tf.newaxis]
    pte = tf.tile(pte, multiples=(1, n_cells_x, n_cells_y, 1))

    ptm = incidence_dict["ptm"]
    ptm = ptm[:, tf.newaxis, tf.newaxis, tf.newaxis]
    ptm = tf.tile(ptm, multiples=(1, n_cells_x, n_cells_y, 1))

    Lx, Ly = L_xy
    ur1 = 1.0  # permeability in reflection region
    ur2 = 1.0  # permeability in transmission region

    ### Step 3: Build convolution matrices for the permittivity and permeability ###
    ERC = convmat(ER_t, PQ[0], PQ[1])
    URC = convmat(UR_t, PQ[0], PQ[1])

    ### Step 4: Wave vector expansion ###
    I = np.eye(np.prod(PQ), dtype=complex)
    I = tf.convert_to_tensor(I, dtype=tf.complex64)
    I = I[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    I = tf.tile(I, multiples=(batch_size, n_cells_x, n_cells_y, n_layers, 1, 1))
    Z = np.zeros((np.prod(PQ), np.prod(PQ)), dtype=complex)
    Z = tf.convert_to_tensor(Z, dtype=tf.complex64)
    Z = Z[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    Z = tf.tile(Z, multiples=(batch_size, n_cells_x, n_cells_y, n_layers, 1, 1))

    k0 = tf.cast(2 * np.pi / lam0, dtype=tf.complex64)
    kinc_x0 = tf.cast(
        refl_n * tf.sin(theta) * tf.cos(phi),
        dtype=tf.complex64,
    )
    kinc_y0 = tf.cast(
        refl_n * tf.sin(theta) * tf.sin(phi),
        dtype=tf.complex64,
    )
    kinc_z0 = tf.cast(refl_n * tf.cos(theta), dtype=tf.complex64)
    kinc_z0 = kinc_z0[:, :, :, 0, :, :]

    # Unit vectors
    # T1 = np.transpose([2 * np.pi / Lx, 0])
    # T2 = np.transpose([0, 2 * np.pi / Ly])
    p_max = np.floor(PQ[0] / 2.0)
    q_max = np.floor(PQ[1] / 2.0)
    p = tf.constant(
        np.linspace(-p_max, p_max, PQ[0]), dtype=tf.complex64
    )  # indices along T1
    p = p[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    p = tf.tile(p, multiples=(1, n_cells_x, n_cells_y, n_layers, 1, 1))
    q = tf.constant(
        np.linspace(-q_max, q_max, PQ[1]), dtype=tf.complex64
    )  # indices along T2
    q = q[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
    q = tf.tile(q, multiples=(1, n_cells_x, n_cells_y, n_layers, 1, 1))

    # Build Kx and Ky matrices
    kx_zeros = tf.zeros(PQ[1], dtype=tf.complex64)
    kx_zeros = kx_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
    ky_zeros = tf.zeros(PQ[0], dtype=tf.complex64)
    ky_zeros = ky_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    kx = kinc_x0 - 2 * np.pi * p / (k0 * Lx) - kx_zeros
    ky = kinc_y0 - 2 * np.pi * q / (k0 * Ly) - ky_zeros

    kx_T = tf.transpose(kx, perm=[0, 1, 2, 3, 5, 4])
    KX = tf.reshape(
        kx_T, shape=(batch_size, n_cells_x, n_cells_y, n_layers, np.prod(PQ))
    )
    KX = tf.linalg.diag(KX)

    ky_T = tf.transpose(ky, perm=[0, 1, 2, 3, 5, 4])
    KY = tf.reshape(
        ky_T, shape=(batch_size, n_cells_x, n_cells_y, n_layers, np.prod(PQ))
    )
    KY = tf.linalg.diag(KY)

    KZref = tf.linalg.matmul(tf.math.conj(ur1 * I), tf.math.conj(er1 * I))
    KZref = KZref - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZref = tf.math.sqrt(KZref)
    KZref = -tf.math.conj(KZref)

    KZtrn = tf.linalg.matmul(tf.math.conj(ur2 * I), tf.math.conj(er2 * I))
    KZtrn = KZtrn - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZtrn = tf.math.sqrt(KZtrn)
    KZtrn = tf.math.conj(KZtrn)

    ### Step 5: Free Space ###
    KZ = I - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZ = tf.math.sqrt(KZ)
    KZ = tf.math.conj(KZ)

    Q_free_00 = tf.linalg.matmul(KX, KY)
    Q_free_01 = I - tf.linalg.matmul(KX, KX)
    Q_free_10 = tf.linalg.matmul(KY, KY) - I
    Q_free_11 = -tf.linalg.matmul(KY, KX)
    Q_free_row0 = tf.concat([Q_free_00, Q_free_01], axis=5)
    Q_free_row1 = tf.concat([Q_free_10, Q_free_11], axis=5)
    Q_free = tf.concat([Q_free_row0, Q_free_row1], axis=4)

    W0_row0 = tf.concat([I, Z], axis=5)
    W0_row1 = tf.concat([Z, I], axis=5)
    W0 = tf.concat([W0_row0, W0_row1], axis=4)

    LAM_free_row0 = tf.concat([1j * KZ, Z], axis=5)
    LAM_free_row1 = tf.concat([Z, 1j * KZ], axis=5)
    LAM_free = tf.concat([LAM_free_row0, LAM_free_row1], axis=4)

    V0 = tf.linalg.matmul(Q_free, tf.linalg.inv(LAM_free))

    ### Step 6: Initialize Global Scattering Matrix ###
    SG = dict({})
    SG_S11 = tf.zeros(shape=(2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=tf.complex64)
    SG["S11"] = expand_and_tile_tf(SG_S11, batch_size, n_cells_x, n_cells_y)

    SG_S12 = tf.eye(num_rows=2 * np.prod(PQ), dtype=tf.complex64)
    SG["S12"] = expand_and_tile_tf(SG_S12, batch_size, n_cells_x, n_cells_y)

    SG_S21 = tf.eye(num_rows=2 * np.prod(PQ), dtype=tf.complex64)
    SG["S21"] = expand_and_tile_tf(SG_S21, batch_size, n_cells_x, n_cells_y)

    SG_S22 = tf.zeros(shape=(2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=tf.complex64)
    SG["S22"] = expand_and_tile_tf(SG_S22, batch_size, n_cells_x, n_cells_y)

    ### Step 7: Calculate eigenmodes ###

    # Build the eigenvalue problem.
    P_00 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
    P_00 = tf.linalg.matmul(P_00, KY)

    P_01 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
    P_01 = tf.linalg.matmul(P_01, KX)
    P_01 = URC - P_01

    P_10 = tf.linalg.matmul(KY, tf.linalg.inv(ERC))
    P_10 = tf.linalg.matmul(P_10, KY) - URC

    P_11 = tf.linalg.matmul(-KY, tf.linalg.inv(ERC))
    P_11 = tf.linalg.matmul(P_11, KX)

    P_row0 = tf.concat([P_00, P_01], axis=5)
    P_row1 = tf.concat([P_10, P_11], axis=5)
    P = tf.concat([P_row0, P_row1], axis=4)

    Q_00 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
    Q_00 = tf.linalg.matmul(Q_00, KY)

    Q_01 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
    Q_01 = tf.linalg.matmul(Q_01, KX)
    Q_01 = ERC - Q_01

    Q_10 = tf.linalg.matmul(KY, tf.linalg.inv(URC))
    Q_10 = tf.linalg.matmul(Q_10, KY) - ERC

    Q_11 = tf.linalg.matmul(-KY, tf.linalg.inv(URC))
    Q_11 = tf.linalg.matmul(Q_11, KX)

    Q_row0 = tf.concat([Q_00, Q_01], axis=5)
    Q_row1 = tf.concat([Q_10, Q_11], axis=5)
    Q = tf.concat([Q_row0, Q_row1], axis=4)

    # Compute eignmodes for the layers in each pixel for the whole batch.
    OMEGA_SQ = tf.linalg.matmul(P, Q)
    LAM, W = eig_general(OMEGA_SQ)
    LAM = tf.sqrt(LAM)
    LAM = tf.linalg.diag(LAM)

    V = tf.linalg.matmul(Q, W)
    V = tf.linalg.matmul(V, tf.linalg.inv(LAM))

    # Scattering matrices for the layers in each pixel for the whole batch.
    W_inv = tf.linalg.inv(W)
    V_inv = tf.linalg.inv(V)
    A = tf.linalg.matmul(W_inv, W0) + tf.linalg.matmul(V_inv, V0)
    B = tf.linalg.matmul(W_inv, W0) - tf.linalg.matmul(V_inv, V0)
    X = tf.linalg.expm(-LAM * k0 * layer_thicknesses)

    S = dict({})
    A_inv = tf.linalg.inv(A)
    S11_left = tf.linalg.matmul(X, B)
    S11_left = tf.linalg.matmul(S11_left, A_inv)
    S11_left = tf.linalg.matmul(S11_left, X)
    S11_left = tf.linalg.matmul(S11_left, B)
    S11_left = A - S11_left
    S11_left = tf.linalg.inv(S11_left)

    S11_right = tf.linalg.matmul(X, B)
    S11_right = tf.linalg.matmul(S11_right, A_inv)
    S11_right = tf.linalg.matmul(S11_right, X)
    S11_right = tf.linalg.matmul(S11_right, A)
    S11_right = S11_right - B
    S["S11"] = tf.linalg.matmul(S11_left, S11_right)

    S12_right = tf.linalg.matmul(B, A_inv)
    S12_right = tf.linalg.matmul(S12_right, B)
    S12_right = A - S12_right
    S12_left = tf.linalg.matmul(S11_left, X)
    S["S12"] = tf.linalg.matmul(S12_left, S12_right)

    S["S21"] = S["S12"]
    S["S22"] = S["S11"]

    # Update the global scattering matrices.
    for l in range(n_layers):
        S_layer = dict({})
        S_layer["S11"] = S["S11"][:, :, :, l, :, :]
        S_layer["S12"] = S["S12"][:, :, :, l, :, :]
        S_layer["S21"] = S["S21"][:, :, :, l, :, :]
        S_layer["S22"] = S["S22"][:, :, :, l, :, :]
        SG = redheffer_star_product(SG, S_layer)

    ### Step 8: Reflection side ###
    # Eliminate layer dimension for tensors as they are unchanging on this dimension.
    KX = KX[:, :, :, 0, :, :]
    KY = KY[:, :, :, 0, :, :]
    KZref = KZref[:, :, :, 0, :, :]
    KZtrn = KZtrn[:, :, :, 0, :, :]
    Z = Z[:, :, :, 0, :, :]
    I = I[:, :, :, 0, :, :]
    W0 = W0[:, :, :, 0, :, :]
    V0 = V0[:, :, :, 0, :, :]

    Q_ref_00 = tf.linalg.matmul(KX, KY)
    Q_ref_01 = ur1 * er1 * I - tf.linalg.matmul(KX, KX)
    Q_ref_10 = tf.linalg.matmul(KY, KY) - ur1 * er1 * I
    Q_ref_11 = -tf.linalg.matmul(KY, KX)
    Q_ref_row0 = tf.concat([Q_ref_00, Q_ref_01], axis=4)
    Q_ref_row1 = tf.concat([Q_ref_10, Q_ref_11], axis=4)
    Q_ref = tf.concat([Q_ref_row0, Q_ref_row1], axis=3)

    W_ref_row0 = tf.concat([I, Z], axis=4)
    W_ref_row1 = tf.concat([Z, I], axis=4)
    W_ref = tf.concat([W_ref_row0, W_ref_row1], axis=3)

    LAM_ref_row0 = tf.concat([-1j * KZref, Z], axis=4)
    LAM_ref_row1 = tf.concat([Z, -1j * KZref], axis=4)
    LAM_ref = tf.concat([LAM_ref_row0, LAM_ref_row1], axis=3)

    V_ref = tf.linalg.matmul(Q_ref, tf.linalg.inv(LAM_ref))

    W0_inv = tf.linalg.inv(W0)
    V0_inv = tf.linalg.inv(V0)
    A_ref = tf.linalg.matmul(W0_inv, W_ref) + tf.linalg.matmul(V0_inv, V_ref)
    A_ref_inv = tf.linalg.inv(A_ref)
    B_ref = tf.linalg.matmul(W0_inv, W_ref) - tf.linalg.matmul(V0_inv, V_ref)

    SR = dict({})
    SR["S11"] = tf.linalg.matmul(-A_ref_inv, B_ref)
    SR["S12"] = 2 * A_ref_inv
    SR_S21 = tf.linalg.matmul(B_ref, A_ref_inv)
    SR_S21 = tf.linalg.matmul(SR_S21, B_ref)
    SR["S21"] = 0.5 * (A_ref - SR_S21)
    SR["S22"] = tf.linalg.matmul(B_ref, A_ref_inv)

    ### Step 9: Transmission side ###
    Q_trn_00 = tf.linalg.matmul(KX, KY)
    Q_trn_01 = ur2 * er2 * I - tf.linalg.matmul(KX, KX)
    Q_trn_10 = tf.linalg.matmul(KY, KY) - ur2 * er2 * I
    Q_trn_11 = -tf.linalg.matmul(KY, KX)
    Q_trn_row0 = tf.concat([Q_trn_00, Q_trn_01], axis=4)
    Q_trn_row1 = tf.concat([Q_trn_10, Q_trn_11], axis=4)
    Q_trn = tf.concat([Q_trn_row0, Q_trn_row1], axis=3)

    W_trn_row0 = tf.concat([I, Z], axis=4)
    W_trn_row1 = tf.concat([Z, I], axis=4)
    W_trn = tf.concat([W_trn_row0, W_trn_row1], axis=3)

    LAM_trn_row0 = tf.concat([1j * KZtrn, Z], axis=4)
    LAM_trn_row1 = tf.concat([Z, 1j * KZtrn], axis=4)
    LAM_trn = tf.concat([LAM_trn_row0, LAM_trn_row1], axis=3)

    V_trn = tf.linalg.matmul(Q_trn, tf.linalg.inv(LAM_trn))

    W0_inv = tf.linalg.inv(W0)
    V0_inv = tf.linalg.inv(V0)
    A_trn = tf.linalg.matmul(W0_inv, W_trn) + tf.linalg.matmul(V0_inv, V_trn)
    A_trn_inv = tf.linalg.inv(A_trn)
    B_trn = tf.linalg.matmul(W0_inv, W_trn) - tf.linalg.matmul(V0_inv, V_trn)

    ST = dict({})
    ST["S11"] = tf.linalg.matmul(B_trn, A_trn_inv)
    ST_S12 = tf.linalg.matmul(B_trn, A_trn_inv)
    ST_S12 = tf.linalg.matmul(ST_S12, B_trn)
    ST["S12"] = 0.5 * (A_trn - ST_S12)
    ST["S21"] = 2 * A_trn_inv
    ST["S22"] = tf.linalg.matmul(-A_trn_inv, B_trn)

    ### Step 10: Compute global scattering matrix ###
    SG = redheffer_star_product(SR, SG)
    SG = redheffer_star_product(SG, ST)

    ### Step 11: Compute source parameters ###

    # Compute mode coefficients of the source.
    delta = np.zeros((batch_size, n_cells_x, n_cells_y, np.prod(PQ)))
    delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1

    # Incident wavevector.
    kinc_x0_pol = tf.math.real(kinc_x0[:, :, :, 0, 0])
    kinc_y0_pol = tf.math.real(kinc_y0[:, :, :, 0, 0])
    kinc_z0_pol = tf.math.real(kinc_z0[:, :, :, 0])
    kinc_pol = tf.concat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], axis=3)

    # Calculate TE and TM polarization unit vectors.
    firstPol = True
    for pol in range(batch_size):
        if kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0:
            ate_pol = np.zeros((1, n_cells_x, n_cells_y, 3))
            ate_pol[:, :, :, 1] = 1
            ate_pol = tf.convert_to_tensor(ate_pol, dtype=tf.float32)
        else:
            # Calculation of `ate` for oblique incidence.
            n_hat = np.zeros((1, n_cells_x, n_cells_y, 3))
            n_hat[:, :, :, 0] = 1
            n_hat = tf.convert_to_tensor(n_hat, dtype=tf.float32)
            kinc_pol_iter = kinc_pol[pol, :, :, :]
            kinc_pol_iter = kinc_pol_iter[tf.newaxis, :, :, :]
            ate_cross = tf.linalg.cross(n_hat, kinc_pol_iter)
            ate_pol = ate_cross / tf.norm(ate_cross, axis=3, keepdims=True)

        if firstPol:
            ate = ate_pol
            firstPol = False
        else:
            ate = tf.concat([ate, ate_pol], axis=0)

    atm_cross = tf.linalg.cross(kinc_pol, ate)
    atm = atm_cross / tf.norm(atm_cross, axis=3, keepdims=True)
    ate = tf.cast(ate, dtype=tf.complex64)
    atm = tf.cast(atm, dtype=tf.complex64)

    # Decompose the TE and TM polarization into x and y components.
    EP = pte * ate + ptm * atm
    EP_x = EP[:, :, :, 0]
    EP_x = EP_x[:, :, :, tf.newaxis]
    EP_y = EP[:, :, :, 1]
    EP_y = EP_y[:, :, :, tf.newaxis]

    esrc_x = EP_x * delta
    esrc_y = EP_y * delta
    esrc = tf.concat([esrc_x, esrc_y], axis=3)
    esrc = esrc[:, :, :, :, tf.newaxis]

    W_ref_inv = tf.linalg.inv(W_ref)

    ### Step 12: Compute reflected and transmitted fields ###
    csrc = tf.linalg.matmul(W_ref_inv, esrc)

    # Compute tranmission and reflection mode coefficients.
    cref = tf.linalg.matmul(SG["S11"], csrc)
    ctrn = tf.linalg.matmul(SG["S21"], csrc)
    eref = tf.linalg.matmul(W_ref, cref)
    etrn = tf.linalg.matmul(W_trn, ctrn)

    rx = eref[:, :, :, 0 : np.prod(PQ), :]
    ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
    tx = etrn[:, :, :, 0 : np.prod(PQ), :]
    ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]

    # Compute longitudinal components.
    KZref_inv = tf.linalg.inv(KZref)
    KZtrn_inv = tf.linalg.inv(KZtrn)
    rz = tf.linalg.matmul(KX, rx) + tf.linalg.matmul(KY, ry)
    rz = tf.linalg.matmul(-KZref_inv, rz)
    tz = tf.linalg.matmul(KX, tx) + tf.linalg.matmul(KY, ty)
    tz = tf.linalg.matmul(-KZtrn_inv, tz)

    ### Step 13: Compute diffraction efficiences ###
    rx2 = tf.math.real(rx) ** 2 + tf.math.imag(rx) ** 2
    ry2 = tf.math.real(ry) ** 2 + tf.math.imag(ry) ** 2
    rz2 = tf.math.real(rz) ** 2 + tf.math.imag(rz) ** 2
    R2 = rx2 + ry2 + rz2
    R = tf.math.real(-KZref / ur1) / tf.math.real(kinc_z0 / ur1)
    R = tf.linalg.matmul(R, R2)
    R = tf.reshape(R, shape=(batch_size, n_cells_x, n_cells_y, PQ[0], PQ[1]))
    REF = tf.math.reduce_sum(R, axis=[3, 4])

    tx2 = tf.math.real(tx) ** 2 + tf.math.imag(tx) ** 2
    ty2 = tf.math.real(ty) ** 2 + tf.math.imag(ty) ** 2
    tz2 = tf.math.real(tz) ** 2 + tf.math.imag(tz) ** 2
    T2 = tx2 + ty2 + tz2
    T = tf.math.real(KZtrn / ur2) / tf.math.real(kinc_z0 / ur2)
    T = tf.linalg.matmul(T, T2)
    T = tf.reshape(T, shape=(batch_size, n_cells_x, n_cells_y, PQ[0], PQ[1]))
    TRN = tf.math.reduce_sum(T, axis=[3, 4])

    return {
        "REF": REF,
        "R": R,
        "TRN": TRN,
        "T": T,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "tx": tx,
        "ty": ty,
        "tz": tz,
    }
