import os
import configparser
import bpy
from blender_utils import *

class CSURUnit:
    
    BASE_DIR = 'F:/Work/csl/roads/CSUR/models/textures/'
    
    #Null is default settings
    NULL = 0
    LANE = 1
    MEDIAN = 2
    BIKE = 3
    CURB = 4
    SIDEWALK = 5
    BARRIER = 6
    
    config = configparser.ConfigParser()
    lane_border = 0

    textures = {'d': []}
    objs = {}

    def initialize(filename):
        config = CSURUnit.config
        config.read(filename)
        CSURUnit.lane_border = float(config['PARAM']['lane_border'])
        # load textures
        texpath = os.path.join(config['PATH']['workdir'], config['PATH']['tex'], config['TEX']['road_d'])
        CSURUnit.textures['d'].append(
            bpy.data.images.load(filepath=texpath)
        )

        #load models:
        for k in config['UNITS'].keys():
            obj = load_unit(os.path.join(config['PATH']['workdir'], config['PATH']['units'], config['UNITS'][k]))
            obj.name = 'CSURUnit_' + k
            CSURUnit.objs[k] = obj
            obj.hide_set(True)

@selection_safe   
def load_unit(path, type=CSURUnit.NULL):
    bpy.ops.import_scene.fbx(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.scale = Vector([1, 1, 1])
    obj.location = Vector([0, 0, 0])
    align(obj.data)
    link_image(obj, CSURUnit.textures['d'][type])
    return obj