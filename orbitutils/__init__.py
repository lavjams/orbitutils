__version__ = "0.1.4"

try:
  __ORBITUTILS_SETUP__
except NameError:
  __ORBITUTILS_SETUP__ = False


if not __ORBITUTILS_SETUP__:
  __all__ = ['OrbitPopulation','OrbitPopulation_FromH5',
             'OrbitPopulation_FromDF','TripleOrbitPopulation',
            'TripleOrbitPopulation_FromDF','TripleOrbitPopulation_FromH5',
             'semimajor']
  
  from .populations import OrbitPopulation,OrbitPopulation_FromH5
  from .populations import TripleOrbitPopulation,TripleOrbitPopulation_FromH5
  from .populations import OrbitPopulation_FromDF,TripleOrbitPopulation_FromDF
  from .populations import BinaryGrid
  
  from .utils import semimajor
