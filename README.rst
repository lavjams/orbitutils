orbitutils
==========
This package allows for easy and very quick Monte Carlo simulation of populations of orbits, giving instantly distributions of projected separations, relative RVs between components, etc.  Includes calculations for eccentric orbits.  Handy for simulating imaging or RV observations.

Makes occasional use of the `astropy <www.astropy.org>`_ ``Quantity`` object to keep track of some quantities, and the astropy ``SkyCoord`` object to keep track of coordinates.  Also uses `pandas <pandas.pydata.org>`_ to save/restore populations.


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

Simulate a population for given primary and secondary mass(es), and orbital periods.  Eccentricity is zero by default, but can be set.

.. code-block:: python

    from orbitutils import OrbitPopulation
    pop = OrbitPopulation(1,1,1000,n=1e4) #Primary mass, secondary mass, orbital period (d)

You can also create a distribution of secondary masses, and/or a distribution of periods:

.. code-block:: python

    import numpy as np
    from orbitutils import OrbitPopulation
    N=1e4
    M2s = np.linspace(0.1,1,N)
    Ps = np.logspace(1,3,N) #periods evenly log-spaced from 10 to 1000 days
    pop = OrbitPopulation(1,M2s,Ps)

``OrbitPopulation`` objects can be saved using the ``save_hdf`` method:

.. code-block:: python

    pop.save_hdf('mypop.h5')

and then read back in/restored as follows:

.. code-block:: python

    from orbitutils import OrbitPopulation_FromH5
    pop_again = OrbitPopulation_FromH5('mypop.h5')

