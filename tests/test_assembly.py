import pytest, os
import numpy as np
import tensorflow as tf
from metabox import rcwa, utils, modeling, assembly

__author__ = "Luocheng Huang"
__copyright__ = "Luocheng Huang"
__license__ = "MIT"


def test_big_fat_test_to_make_sure_it_runs():
    # Now let's define some materials for our simulations
    TiO2 = rcwa.Material("Si3N4")
    quartz = rcwa.Material("quartz")

    # Define sampling wavelengths, here we only simulate 3 wavelength for simplicity
    # In practice, you may want to simulate more wavelengths for better sampling density.
    incidence = utils.Incidence(wavelength=np.linspace(460e-9, 700e-9, 1))

    # Define the unit cell periodicity
    periodicity = (350e-9, 350e-9)

    # Define RCWA simulation configuration
    sim_config = rcwa.SimConfig(
        xy_harmonics=(7, 7),  # Fourier orders in x and y
        resolution=256,  # grid resolution per periodicity
        return_tensor=True,  # return tensor instead of a SimulationResult object
        minibatch_size=10,  # number of simulations to run in parallel
    )

    # Instantiate a `Feature` to parameterize the width of the square
    width = utils.Feature(
        vmin=0, vmax=periodicity[0], name="radius", sampling=10
    )
    square = rcwa.Rectangle(material=TiO2, x_width=width, y_width=width)
    patterned_layer = rcwa.Layer(material=1, thickness=800e-9, shapes=[square])
    substrate = rcwa.Layer(material=quartz, thickness=1000e-9)
    cell = rcwa.UnitCell(
        layers=[patterned_layer, substrate],
        periodicity=periodicity,
    )
    protocell = rcwa.ProtoUnitCell(cell)

    sim_lib = modeling.sample_protocell(
        protocell=protocell,
        incidence=incidence,
        sim_config=sim_config,
    )

    # make temp folder
    os.makedirs("./tests/temp", exist_ok=True)
    sim_lib.save("TiO2_square_sim_lib", "./tests/temp", overwrite=True)
    # Now that we have saved our simulations, we can load them back in
    # and use them to train a metamodel neural network.
    loaded_sim_lib = modeling.load_simulation_library(
        "TiO2_square_sim_lib", "./tests/temp"
    )

    # Let's train a DNN with the following architecture:
    # Input layer -> 10 (relu) -> 128 (tanh) -> 256 (relu) -> 256 (tanh) -> 128 (relu) -> 10 (tanh) -> Output layer
    model = modeling.create_and_train_model(
        loaded_sim_lib,
        n_epochs=100,
        hidden_layer_units_list=[10, 128, 10],
        activation_list=["relu", "tanh", "tanh"],
        train_batch_size=100,
    )
    model.save("TiO2_square_metamodel", "./tests/temp", overwrite=True)

    # Load the metamodel we created in tutorial 1.
    metamodel = modeling.load_metamodel(
        "TiO2_square_metamodel", "./tests/temp"
    )
    # Define the bounds of the feature.
    metamodel.set_feature_constraint("width", vmin=10e-9, vmax=330e-9)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=10e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        metamodel=metamodel,  # the metamodel to use
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
    )

    # Define the incidence wavelengths and angles.
    incidence = assembly.Incidence(
        wavelength=np.linspace(400e-9, 700e-9, 2),
        phi=[0],  # normal incidence
        theta=[0],  # normal incidence
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[metasurface],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,  # Define the figure of merit.
    )

    # Use the Adam optimizer to optimize the lens assembly. This rate should be
    # empirically determined.
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-8)

    # Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
    history = assembly.optimize_single_lens_assembly(
        lens_assembly,
        optimizer,
        n_iter=5,
        verbose=1,
        keep_best=True,
    )
