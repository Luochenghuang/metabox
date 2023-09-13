import pytest, os
import numpy as np
import tensorflow as tf
from metabox import rcwa, utils, modeling, assembly, propagation
from metabox.assembly import AtomArray2D, AtomArray1D, CustomFigureOfMerit

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
        minibatch_size=4,  # number of simulations to run in parallel
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
        n_epochs=2,
        hidden_layer_units_list=[10, 10],
        activation_list=["relu", "tanh"],
        train_batch_size=100,
    )
    model_2 = modeling.create_and_train_model(
        loaded_sim_lib,
        n_epochs=2,
        hidden_layer_units_list=[10, 10],
        activation_list=["relu", "tanh"],
        train_batch_size=100,
        limit_output_to_unity=True,
    )
    model.plot_training_history()
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

    # custom figure of merit
    fom = assembly.CustomFigureOfMerit("reduce_sum(strehl_ratio)")
    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[
            metasurface,
            metasurface_2,
        ],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=fom,  # Define the figure of merit.
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
    assert isinstance(history, list)
    for item in history:
        assert np.isnan(item) == False

    assembly.save_lens_assembly(
        lens_assembly, "TiO2_square_lens", "./tests/temp", overwrite=True
    )
    assembly.load_lens_assembly("TiO2_square_lens", "./tests/temp")

    metasurface.show_feature_map()


class TestAtomArray2D:
    @pytest.fixture
    def valid_tensor(self):
        return tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

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

    def test_set_feature_map_metamodel(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray2D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        atom_array.set_feature_map("radius", np.ones((3, 3)))
        assert atom_array.tensor.shape == (1, 9)

    def test_set_feature_map_ProtoUnitCell(
        self, valid_tensor, valid_proto_unit_cell
    ):
        atom_array = AtomArray2D(
            valid_tensor, period=1.0, proto_unit_cell=valid_proto_unit_cell
        )
        atom_array.set_feature_map("radius", np.ones((3, 3)))
        assert atom_array.tensor.shape == (1, 9)

    def test_set_to_use_rcwa(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray2D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        atom_array.set_to_use_rcwa()
        assert not atom_array.use_mmodel


class TestAtomArray1D:
    @pytest.fixture
    def valid_tensor(self):
        return tf.constant([[1, 2, 3, 4, 5, 6]])

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

    def test_set_feature_map(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray1D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        atom_array.set_feature_map("radius", np.ones((1, 6)))
        assert atom_array.tensor.shape == (1, 6)

    def test_set_feature_map_puc(self, valid_tensor, valid_proto_unit_cell):
        atom_array = AtomArray1D(
            valid_tensor, period=1.0, proto_unit_cell=valid_proto_unit_cell
        )
        atom_array.set_feature_map("radius", np.ones((1, 6)))
        assert atom_array.tensor.shape == (1, 6)

    def test_set_to_use_rcwa(self, valid_tensor, valid_mmodel):
        atom_array = AtomArray1D(valid_tensor, period=1.0, mmodel=valid_mmodel)
        atom_array.set_to_use_rcwa()
        assert not atom_array.use_mmodel


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


class TestAmplitudeMask:
    def test_post_init(self):
        mask = assembly.AmplitudeMask(
            4.0, 1.0, 1.0, 1.0, 1.0, True, True, False
        )
        assert mask.n_pixels_radial == 2
        assert mask.propagator_cache == (None, None)
        assert mask.variables == []

    def test_optimizer_hook(self):
        mask = assembly.AmplitudeMask(
            4.0, 1.0, 1.0, 1.0, 1.0, True, True, False, 0.5
        )
        mask.optimizer_hook()
        assert mask.threshold_param == 1.5

    def test_get_modulation_2d(self):
        mask = assembly.AmplitudeMask(
            4.0, 1.0, 1.0, 1.0, 1.0, True, True, False
        )
        incidence = utils.Incidence(wavelength=[1])
        result = mask.get_modulation_2d(incidence)
        assert isinstance(result, tf.Tensor)
        assert result.shape == (1, 4, 4)

    def test_get_end_field(self):
        mask = assembly.AmplitudeMask(
            4.0, 1.0, 1.0, 1.0, 1.0, True, True, False
        )
        incidence = utils.Incidence(wavelength=[1], theta=[0], phi=[0])
        incident_field = propagation.Field2D(
            4,
            [1],
            [0],
            [0],
            1,
            True,
            True,
            True,
            tf.cast(tf.ones((1, 4, 4)), tf.complex64),
        )
        result = mask.get_end_field(incidence, incident_field, 1.0)
        assert isinstance(result, propagation.Field2D)


def test_RefractiveEvenAsphere():
    incidence = assembly.Incidence(
        wavelength=[700e-9],
        theta=[0],
        phi=[0],
    )

    asph1 = assembly.RefractiveEvenAsphere(
        diameter=0.1e-3,
        refractive_index=1.5131,
        thickness=0.1e-3,
        periodicity=0.25e-6,
        init_coeff=[6.124, 97.618],
    )

    asph2 = assembly.RefractiveEvenAsphere(
        diameter=0.2e-3,
        refractive_index=1.0,
        thickness=0.1e-3,
        periodicity=0.25e-6,
        init_coeff=[0.731, -189.048],
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[asph1, asph2],
        incidence=incidence,  # Define the incidence.
    )

    asph1.show_sag()
    end_field = lens_assembly.compute_field_on_sensor()
    assert isinstance(end_field, propagation.Field2D)


def test_Binary2():
    incidence = assembly.Incidence(
        wavelength=[700e-9],
        theta=[0, 5, 10],
        phi=[0],
    )

    # Define the binary surface with the coefficients.
    bin1 = assembly.Binary2(
        diameter=40e-6,
        refractive_index=1.0,
        thickness=0.1e-3,
        periodicity=10e-6,
        init_coeff=[6.763, 2.1, -0.032, 0.055, -7.122e-3],
    )

    bin2 = assembly.Binary2(
        diameter=40e-6,
        refractive_index=1.0,
        thickness=0.2e-3,
        periodicity=10e-6,
        init_coeff=[-249.003, -10.79, -0.16, 0.24, 0.12],
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[bin1, bin2],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
    )

    end_field = lens_assembly.compute_field_on_sensor()
    assert isinstance(end_field, propagation.Field2D)


class TestCustomFOM:
    def test_initialization(self):
        # Test that the class is initialized properly with valid inputs
        fom = CustomFigureOfMerit("5 + strehl_ratio")
        assert fom.expression == "5 + strehl_ratio"

        # Test that the class raises an error if the expression is None
        with pytest.raises(ValueError):
            fom = CustomFigureOfMerit(None)

    def test_get_validation_errors(self):
        # Test that the method raises an error for invalid characters in the expression
        with pytest.raises(ValueError) as excinfo:
            fom = CustomFigureOfMerit("psf + strehl_ratios")
        assert "Ensure you are using" in str(excinfo.value)

        # Test that the method does not raise an error for valid characters in the expression
        fom = CustomFigureOfMerit("psf + strehl_ratio")
        errors = fom.get_validation_errors()
        assert len(errors) == 0

    def test_is_valid_expression(self):
        # Test that the method returns True for a valid expression
        fom = CustomFigureOfMerit("reduce_sum(psf)")
        assert fom.is_valid_expression(fom.expression)

    def test_custom_data(self):
        # Test that the class handles custom data properly
        data = {
            "r_targ": tf.ones((1, 2, 2)),
            "g_targ": tf.ones((1, 2, 2)),
            "b_targ": tf.ones((1, 2, 2)),
        }
        expression = "-log(r_targ.dist(psf[0, :, :]) + g_targ.dist(psf[1, :, :]) + b_targ.dist(psf[2, :, :]))"
        fom = CustomFigureOfMerit(expression, data)
        assert "r_targ" in fom.data
        assert fom.is_valid_expression(fom.expression)


def test_cartesian_distance():
    # Define the inputs
    psf = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    intensity_target = assembly.IntensityTarget(psf * 2, crop_factor=0.9)

    # Call the function
    distance = assembly.cartesian_distance(intensity_target, psf)

    # Check that the returned value is a tensor.
    assert isinstance(distance, tf.Tensor)

    # Check that the returned value is a scalar.
    assert distance.shape == ()

    # Check that the returned value is non-negative.
    assert distance.numpy() >= 0


def test_optimize_multiple_LA():
    metamodel = modeling.load_metamodel(
        "TiO2_square_metamodel", "./tests/temp"
    )

    # Create a metasurface.
    metasurface = assembly.Metasurface(
        diameter=4e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=200e-6,  # the distance to the next surface
        metamodel=metamodel,  # the metamodel to use
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
    )

    # Define the incidence wavelengths and angles.
    incidence = assembly.Incidence(
        wavelength=np.linspace(
            0.4e-6, 0.6e-6, 2
        ),  # 11 wavelengths between 400 and 800 nm
        phi=[0, 5],  # normal incidence
        theta=[0, 5],  # normal incidence
    )

    # Create a lens assembly.
    batched_lens_assembly = assembly.LensAssembly(
        surfaces=[metasurface],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,  # Define the figure of merit.
        use_padding=True,
    )

    unbatched_assemblies = assembly.unbatch_lens_assembley(
        batched_lens_assembly
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-9)
    optimizer.build(batched_lens_assembly.get_variables())
    history = assembly.optimize_multiple_lens_assemblies(
        unbatched_assemblies, optimizer, n_iter=3, verbose=1
    )
    assert isinstance(history, list)


def test_optimization_of_proto_unit_cell_metasurface():
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
        diameter=10e-6,  # 100 microns in diameter
        refractive_index=1.0,  # the propagation medium after the metasurface
        thickness=30e-6,  # the distance to the next surface
        proto_unit_cell=protocell,
        enable_propagator_cache=True,  # cache the propagators for faster computation
        set_structures_variable=True,  # set the structures as a variable to optimize
    )

    # Define the incidence wavelengths and angles.
    incidence = assembly.Incidence(
        wavelength=np.linspace(400e-9, 700e-9, 1),
        phi=[0],  # normal incidence
        theta=[0],  # normal incidence
    )

    # Create a lens assembly.
    lens_assembly = assembly.LensAssembly(
        surfaces=[metasurface],  # Define the array of surfaces. Here only one.
        incidence=incidence,  # Define the incidence.
        figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,  # Define the figure of merit.
        use_x_pol=True,  # Use the x-polarization.
    )

    # Use the Adam optimizer to optimize the lens assembly. This rate should be
    # empirically determined.
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-8)
    optimizer.build(lens_assembly.get_variables())

    # Optimize the lens assembly. Returns the best-optimized lens assembly and the loss history.
    history = assembly.optimize_single_lens_assembly(
        lens_assembly,
        optimizer,
        n_iter=2,
        verbose=1,
        keep_best=True,
    )
    assert isinstance(history, list)
    for h in history:
        assert np.isnan(h) == False
