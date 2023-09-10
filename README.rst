.. figure:: images/metabox.svg
   :width: 500
   :alt: metabox logo
   :align: center

.. image:: https://github.com/Luochenghuang/metabox/actions/workflows/ci.yml/badge.svg
   :alt: Built Status
   :target: https://https://github.com/Luochenghuang/metabox/actions/workflows/ci.yml
.. image:: https://readthedocs.org/projects/metabox/badge/?version=latest
   :alt: ReadTheDocs
   :target: https://metabox.readthedocs.io/en/latest/
.. image:: https://coveralls.io/repos/github/Luochenghuang/metabox/badge.svg?branch=main
   :target: https://coveralls.io/github/Luochenghuang/metabox?branch=main

==========================================================================
``metabox``: A High-Level Python API for Diffractive Optical System Design
==========================================================================

    metabox is a Python package built on TensorFlow, enabling the design, evaluation and optimization of complex diffractive optical systems with ease, flexibility, and high performance.

`metabox` is a high-level Python package specifically designed for the creation, evaluation, and inverse optimization of diffractive optical systems. Leaning on the robust capabilities of TensorFlow, `metabox` offers a comprehensive and user-friendly API for optical system design.

The package is built with flexibility at its core, making it easy to add new components, define custom merit functions, and employ various optimization algorithms. It's designed to be highly performant and scalable, capable of managing systems with millions of degrees of freedom. With its intuitive structure, `metabox` facilitates the design of intricate diffractive optical systems with minimal lines of code.

Key features of `metabox` include:

- A `rcwa` solver, derived from rcwa_tf[1], for direct computation of meta-atoms' diffraction efficiency.
- A built-in `raster` module for parameterizing meta-atoms' features.
- An easy-to-use sampling system for features, which can train a metamodel to replace the `rcwa` solver, thus significantly speeding up simulations and optimization processes.
- A module for sequential optics to model light propagation through the optical system.
- An `assembly` module offering a suite of tools for building the optical system from meta-atoms, apertures, and other optical components.
- A `merit` module for evaluating and inverse-designing the performance of the optical system.
- An `rcwa.Material` class for accessing pre-defined materials and their optical properties.
- An `export` module that allows for the export of the diffractive optical design to a `.gds` file for fabrication.

Overall, `metabox` is a powerful tool for both beginners and experienced users in the field of optical system design. By simplifying and accelerating the design process, it paves the way for innovative developments in the optical industry.

[1] Colburn, S., Majumdar, A. Inverse design and flexible parameterization of meta-optics using algorithmic differentiation. Commun Phys 4, 65 (2021).

=======
Install
=======
Run the following commands to install `metabox`::

    git clone https://github.com/Luochenghuang/metabox.git
    cd metabox
    pip install .

===============
Getting Started
===============

`Tutorial 1: Metamodeling <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_1_metamodeling.ipynb>`_

`Tutorial 2: Lens Optimization and Exporting <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_2_lens_optimization.ipynb>`_

`Tutorial 3: Optimization Serialization
<https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_3_optimization_serialization.ipynb>`_

`Tutorial 4: Zemax Binary2 Import <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_4_binary2_zemax.ipynb>`_

`Tutorial 5: Refractive Surfaces Simulation <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_5_refractive_surfaces.ipynb>`_

`Tutorial 6: Refractive Surfaces Optimization <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_6_optimize_refractive.ipynb>`_

`Tutorial 7: Hologram Optimization <https://github.com/Luochenghuang/metabox/blob/main/examples/tutorial_7_holograms.ipynb>`_

=============
Documentation
=============

`Module Reference <https://metabox.readthedocs.io/en/latest/api/modules.html>`_

`Home Page <https://metabox.readthedocs.io/en/latest/>`_

============
Contributors
============

* Luocheng Huang <luocheng@uw.edu>

==================
Citing ``metabox``
==================

The manuscript is in preparation.

=============================
Making Changes & Contributing
=============================

This project uses `pre-commit`, please make sure to install it before making any
changes::

    pip install pre-commit
    cd metabox
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
