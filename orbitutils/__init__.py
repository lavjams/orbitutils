__version__ = "0.1.3"

try:
  __ORBITUTILS_SETUP__
except NameError:
  __ORBITUTILS_SETUP__ = False


if not __ORBITUTILS_SETUP__:
  __all__ = ['OrbitPopulation','OrbitPopulation_FromH5','TripleOrbitPopulation',
            'TripleOrbitPopulation','TripleOrbitPopulation_FromH5','semimajor']
  
  from populations import OrbitPopulation,OrbitPopulation_FromH5
  from populations import TripleOrbitPopulation,TripleOrbitPopulation_FromH5
  from populations import BinaryGrid
  
  from utils import semimajor
