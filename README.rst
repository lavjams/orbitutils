orbitutils
==========

This package allows for easy and very quick Monte Carlo simulation of populations of orbits, giving instantly distributions of projected separations, relative RVs between components, etc.  Includes calculations for eccentric orbits.

See below for a quick intro, and the `notebook demo <http://nbviewer.ipython.org/github/timothydmorton/orbitutils/blob/master/notebooks/demo.ipynb>`_ for more.

Installation
------------

::

   $ pip install [--user] orbitutils
   
Or clone the repository and install:

::

    $ git clone https://github.com/timothydmorton/orbitutils.git
    $ cd orbitutils
    $ python setup.py install [--user]

Basic usage
-----------

Simulate a population for given primary and secondary mass(es), and orbital periods.  Eccentricity is zero by default.

.. code-block:: python

    from orbitutils import OrbitPopulation
    pop = OrbitPopulation(1,1,1000,n=1e4) #Primary mass, secondary mass, orbital period (d)
    
