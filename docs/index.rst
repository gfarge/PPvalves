.. PPvalves documentation master file, created by
   sphinx-quickstart on Thu Feb 25 11:38:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PPvalves's documentation!
====================================

**Pore pressure valves** (**PPvalves** or *PPv*) is a Python framework to implement a
model of fluid pressure circulation and dynamic permeability in a geological
plumbing system. Details on the physical model, numerical implementation and
how it can be used can be found in `this preprint <https://doi.org/10.31223/X59G7P>`_. If you find PPvalves useful, please consider citing it.

PPvalves numerically solves the diffusion of pore pressure in a 1D permeable conduit, in
which valves of low permeability can open and close in response to the local pore pressure field.
Such setup is meant to explore the interactions between valves, which can be thought of as an 
emulation of elementary seismic sources.

The package also contains a variety of functions to analyse the synthetic
seismicity catalogs which are produced, to rapidly plot specific figures, to
compare numerical solutions to theoretical results...


Table of contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation>
   Basic workflow <workflow>
   Examples <examples>
   Modules <api/modules>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
