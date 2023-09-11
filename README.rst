.. figure:: https://github.com/Luochenghuang/metabox/blob/main/images/metabox.svg
   :width: 500
   :alt: metabox logo
   :align: center

.. image:: https://badge.fury.io/py/metabox.svg
    :target: https://badge.fury.io/py/metabox
.. image:: https://github.com/Luochenghuang/metabox/actions/workflows/ci.yml/badge.svg
   :alt: Built Status
   :target: https://github.com/Luochenghuang/metabox/actions/workflows/ci.yml
.. image:: https://readthedocs.org/projects/metabox/badge/?version=latest
   :alt: ReadTheDocs
   :target: https://metabox.readthedocs.io/en/latest/
.. image:: https://coveralls.io/repos/github/Luochenghuang/metabox/badge.svg?branch=main
   :target: https://coveralls.io/github/Luochenghuang/metabox?branch=main
.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://github.com/Luochenghuang/metabox/blob/main/LICENSE.txt

==========================================================================
``metabox``: A High-Level Python API for Diffractive Optical System Design
==========================================================================

    metabox is a Python package built on TensorFlow, enabling the design, evaluation and optimization of complex diffractive optical systems with ease, flexibility, and high performance.

``metabox`` is a high-level Python package specifically designed for the creation, evaluation, and inverse optimization of diffractive optical systems. Leaning on the robust capabilities of TensorFlow, ``metabox`` offers a comprehensive and user-friendly API for optical system design.

The package is built with flexibility at its core, making it easy to add new components, define custom merit functions, and employ various optimization algorithms. It's designed to be highly performant and scalable, capable of managing systems with millions of degrees of freedom. With its intuitive structure, ``metabox`` facilitates the design of intricate diffractive optical systems with minimal lines of code.

Key features of ``metabox`` include:

- A ``rcwa`` solver, derived from `rcwa_tf <https://github.com/scolburn54/rcwa_tf>`_, for direct computation of meta-atoms' diffraction efficiency.
- A built-in ``raster`` module for parameterizing meta-atoms' features.
- An easy-to-use sampling system for features, which can train a metamodel to replace the ``rcwa`` solver, thus significantly speeding up simulations and optimization processes.
- A module for sequential optics to model light propagation through the optical system.
- An ``assembly`` module offering a suite of tools for building the optical system from meta-atoms, apertures, and other optical components.
- A ``merit`` module for evaluating and inverse-designing the performance of the optical system.
- An ``rcwa.Material`` class for accessing pre-defined materials and their optical properties.
- An ``export`` module that allows for the export of the diffractive optical design to a ``.gds`` file for fabrication.

Overall, ``metabox`` is a powerful tool for both beginners and experienced users in the field of optical system design. By simplifying and accelerating the design process, it paves the way for innovative developments in the optical industry.

=======
Install
=======

Install ``metabox`` via ``pip``::

    pip install metabox

===============
Getting Started
===============
Try out ``metabox`` for free on Google Colab. Here are some tutorials on Colab. You can find the local versions `here <https://github.com/Luochenghuang/metabox/tree/main/examples>`_.

`Tutorial 1: Metamodeling <https://colab.research.google.com/drive/12DW9yZPtM90IO_DeU393wANLnnsgXMrM?authuser=1>`_

`Tutorial 2: Lens Optimization and Exporting <https://colab.research.google.com/drive/1dazKEjwD4f-65AOmrykuM2LLKpb_mz2Y?authuser=1>`_

`Tutorial 3: Optimization Serialization
<https://colab.research.google.com/drive/1dfKwsOwsaqMLDy2ibaREksEbGFp4diKZ?authuser=1>`_

`Tutorial 4: Zemax Binary2 Import <https://colab.research.google.com/drive/1iOliSeB_Cg2XgjP1GgIXKJBqWoRthIMt?authuser=1>`_

`Tutorial 5: Refractive Surfaces Simulation <https://colab.research.google.com/drive/1-16cP5P-OgjarXQnzieOBffGKcfJ_Zs5?authuser=1>`_

`Tutorial 6: Refractive Surfaces Optimization <https://colab.research.google.com/drive/1l1ekS4xEpvMIz_JPv-K4skFKQhsLBHdA?authuser=1>`_

`Tutorial 7: Hologram Optimization <https://colab.research.google.com/drive/1-jX9WEyNQYG5klSog5ULoiN6jcXi5X5l?authuser=1>`_

=============
Documentation
=============

`Module Reference <https://metabox.readthedocs.io/en/latest/api/modules.html>`_

`Home Page <https://metabox.readthedocs.io/en/latest/>`_

============
Contributors
============

* Luocheng Huang: luocheng@uw.edu, https://github.com/Luochenghuang

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

To make an editable installation, run the following commands::

    git clone https://github.com/Luochenghuang/metabox.git
    cd metabox
    pip install -e .
