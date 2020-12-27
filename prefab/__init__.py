import importlib
try:
    from .assetmaker import make
except ImportError:
    print("Blender not found, assetmaker will not be imported")

