import os, sys

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKDIR)

from core.csur import LANEWIDTH as LW
from core.assets import Asset
from modeling import Modeler

asset = Asset(LW/2, 4, LW/2, [2,3])

mode = 'e'

seg = asset.get_model(mode)

modeler = Modeler(os.path.join(WORKDIR, "csur.ini"))
modeler.make(seg, mode)

