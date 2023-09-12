import pytest, os
import numpy as np
import tensorflow as tf
from metabox import rcwa, utils, modeling, assembly, propagation
from metabox.assembly import AtomArray2D, AtomArray1D

__author__ = "Luocheng Huang"
__copyright__ = "Luocheng Huang"
__license__ = "MIT"


def test_rcwa():
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
        xy_harmonics=(3, 3),  # Fourier orders in x and y
        resolution=64,  # grid resolution per periodicity
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


def test_metamodel():
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


def test_big_fat_test_to_make_sure_it_runs():
    # Load the metamodel we created in tutorial 1.
    metamodel = modeling.load_metamodel(
        "TiO2_square_metamodel", "./tests/temp"
    )
    # Define the bounds of the feature.
    metamodel.set_feature_constraint("radius", vmin=10e-9, vmax=330e-9)

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=5e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        metamodel=metamodel,  # the metamodel to use
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
        use_circular_expansions=True,
    )
    metasurface_2 = assembly.Metasurface(
        diameter=5e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        metamodel=metamodel,  # the metamodel to use
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
        use_circular_expansions=False,
    )

    # Define the incidence wavelengths and angles.
    incidence = assembly.Incidence(
        wavelength=np.linspace(400e-9, 700e-9, 2),
        phi=[0],  # normal incidence
        theta=[0],  # normal incidence
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[
            metasurface,
            metasurface_2,
        ],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,  # Define the figure of merit.
    )

    # Use the Adam optimizer to optimize the lens assembly. This rate should be
    # empirically determined.
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-8)
    optimizer.build(lens_assembly.get_variables())

    # Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
    history = assembly.optimize_single_lens_assembly(
        lens_assembly,
        optimizer,
        n_iter=5,
        verbose=1,
        keep_best=True,
    )

    assembly.save_lens_assembly(
        lens_assembly, "TiO2_square_lens", "./tests/temp", overwrite=True
    )
    assembly.load_lens_assembly("TiO2_square_lens", "./tests/temp")

    metasurface.show_feature_map()


class TestAtomArray2D:
    @pytest.fixture
    def valid_tensor(self):
        return tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])

    @pytest.fixture
    def valid_mmodel(self):
        return modeling.load_metamodel("TiO2_square_metamodel", "./tests/temp")

    @pytest.fixture
    def valid_proto_unit_cell(self):
        # Now let's define some materials for our simulations
        TiO2 = rcwa.Material("Si3N4")
        quartz = rcwa.Material("quartz")

        # Define the unit cell periodicity
        periodicity = (350e-9, 350e-9)

        # Instantiate a `Feature` to parameterize the width of the square
        width = utils.Feature(
            vmin=0, vmax=periodicity[0], name="radius", sampling=10
        )
        square = rcwa.Rectangle(material=TiO2, x_width=width, y_width=width)
        patterned_layer = rcwa.Layer(
            material=1, thickness=800e-9, shapes=[square]
        )
        substrate = rcwa.Layer(material=quartz, thickness=1000e-9)
        cell = rcwa.UnitCell(
            layers=[patterned_layer, substrate],
            periodicity=periodicity,
        )
        protocell = rcwa.ProtoUnitCell(cell)
        return protocell

    def test_init_with_mmodel(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray2D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        assert atom_array.use_mmodel

    def test_init_with_proto_unit_cell(
        self, valid_tensor, valid_proto_unit_cell
    ):
        atom_array = AtomArray2D(
            valid_tensor, period=1.0, proto_unit_cell=valid_proto_unit_cell
        )
        assert not atom_array.use_mmodel

    def test_init_with_both_mmodel_and_proto_unit_cell(
        self, valid_tensor, valid_mmodel, valid_proto_unit_cell
    ):
        with pytest.raises(ValueError):
            AtomArray2D(
                valid_tensor,
                period=1.0,
                mmodel=valid_mmodel,
                proto_unit_cell=valid_proto_unit_cell,
            )

    def test_init_without_mmodel_and_proto_unit_cell(self, valid_tensor):
        with pytest.raises(ValueError):
            AtomArray2D(valid_tensor, period=1.0)

    def test_find_feature_valid_input(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray2D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        index = atom_array.find_feature_index("radius")
        assert index == 0

    def test_find_feature_invalid_input(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray2D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        with pytest.raises(ValueError):
            atom_array.find_feature_index(
                "Without music, life would be a mistake."
            )


class TestAtomArray1D:
    @pytest.fixture
    def valid_tensor(self):
        return tf.constant([1, 2, 3, 4, 5, 6])

    @pytest.fixture
    def valid_mmodel(self):
        return modeling.load_metamodel("TiO2_square_metamodel", "./tests/temp")

    @pytest.fixture
    def valid_proto_unit_cell(self):
        # Now let's define some materials for our simulations
        TiO2 = rcwa.Material("Si3N4")
        quartz = rcwa.Material("quartz")

        # Define the unit cell periodicity
        periodicity = (350e-9, 350e-9)

        # Instantiate a `Feature` to parameterize the width of the square
        width = utils.Feature(
            vmin=0, vmax=periodicity[0], name="radius", sampling=10
        )
        square = rcwa.Rectangle(material=TiO2, x_width=width, y_width=width)
        patterned_layer = rcwa.Layer(
            material=1, thickness=800e-9, shapes=[square]
        )
        substrate = rcwa.Layer(material=quartz, thickness=1000e-9)
        cell = rcwa.UnitCell(
            layers=[patterned_layer, substrate],
            periodicity=periodicity,
        )
        protocell = rcwa.ProtoUnitCell(cell)
        return protocell

    def test_init_with_mmodel(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray1D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        assert atom_array.use_mmodel

    def test_init_with_proto_unit_cell(
        self, valid_tensor, valid_proto_unit_cell
    ):
        atom_array = AtomArray1D(
            valid_tensor, period=1.0, proto_unit_cell=valid_proto_unit_cell
        )
        assert not atom_array.use_mmodel

    def test_init_with_both_mmodel_and_proto_unit_cell(
        self, valid_tensor, valid_mmodel, valid_proto_unit_cell
    ):
        with pytest.raises(ValueError):
            AtomArray2D(
                valid_tensor,
                period=1.0,
                mmodel=valid_mmodel,
                proto_unit_cell=valid_proto_unit_cell,
            )

    def test_init_without_mmodel_and_proto_unit_cell(self, valid_tensor):
        with pytest.raises(ValueError):
            AtomArray1D(valid_tensor, period=1.0)

    def test_find_feature_valid_input(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray1D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        index = atom_array.find_feature_index("radius")
        assert index == 0

    def test_find_feature_invalid_input(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray1D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        with pytest.raises(ValueError):
            atom_array.find_feature_index(
                "That which does not kill us makes us stronger."
            )


def test_surface():
    s = assembly.Surface(1.0, 1.5, 0.1)
    assert s.diameter == 1.0
    assert s.refractive_index == 1.5
    assert s.thickness == 0.1
    assert s.get_penalty() == 0.0


def test_aperture():
    a = assembly.Aperture(1.0, 1.5, 0.1, 0.01)
    assert a.diameter == 1.0
    assert a.refractive_index == 1.5
    assert a.thickness == 0.1
    assert a.periodicity == 0.01
    assert a.enable_propagator_cache == False
    assert a.store_end_field == False
    assert a.n_pixels_radial == 50


def test_get_modulation_2d():
    a = assembly.Aperture(1.0, 1.5, 0.1, 0.01)
    incidence = utils.Incidence(wavelength=[500e-9], theta=[0], phi=[0])
    mod_2d = a.get_modulation_2d(incidence)
    assert mod_2d.shape == (1, 100, 100)


def test_get_end_field():
    a = assembly.Aperture(1.0, 1.5, 0.1, 0.01)
    incidence = utils.Incidence(wavelength=[500e-9], theta=[0], phi=[0])
    incident_field = propagation.Field2D(
        tensor=np.ones((1, 100, 100)),
        period=0.01,
        n_pixels=100,
        wavelength=[500e-9],
        theta=[0],
        phi=[0],
        upsampling=1,
        use_antialiasing=True,
        use_padding=True,
    )
    end_field = a.get_end_field(
        incidence, incident_field, 1.0, use_padding=True, use_x_pol=True
    )
    assert end_field.tensor.shape == (1, 100, 100)
