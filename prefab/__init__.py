import importlib
try:
    from .assetmaker import make
except ImportError as e:
    print("The following error during Blender import is encountered, assetmaker will not be imported")
    print(e)

