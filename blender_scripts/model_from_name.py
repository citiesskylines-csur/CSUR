import os, sys

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKDIR)

from modeling import Modeler
from prefab.compiler import asset_from_name

asset = asset_from_name("4R=2R2R4P")

modeler = Modeler(os.path.join(WORKDIR, "csur.ini"))

modeler.make(asset.get_model('e'), mode='e')


