__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '1.4.0'

from .map import MapSamples
from .tree import QuadTree

__all__ = ['QuadTree', 'MapSamples']
