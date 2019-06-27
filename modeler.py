import configparser
import os
import bpy
from mathutils import Vector
from blender_utils import *
import csur
from csur import Segment, CSURFactory

LANEWIDTH = csur.LANEWIDTH

'''
Creates CSUR models.
'''
class Modeler:
    
    '''
    Constants for unit types
    Null is the default modeling settings; different from Segment.EMPTY
    '''
    NULL = 0
    SIDEWALK = 1
    WALL = 2

    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        # lane_border indicates the width that a lane unit extends at the edge
        self.lane_border = float(self.config['PARAM']['lane_border'])
        # load textures
        self.texpath = os.path.join(self.config['PATH']['workdir'], 
                                    self.config['PATH']['tex'])
        # road_d is the Default texture map
        self.textures = {'d':{}}
        self.textures['d'][Modeler.NULL] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['road_d']))
        self.textures['d'][Modeler.SIDEWALK] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['sidewalk_d']))
        # wall has the same texture as sidewalk
        self.textures['d'][Modeler.WALL] = self.textures['d'][Modeler.SIDEWALK]

        #load models:
        self.objs = {}
        for k in self.config['UNITS'].keys():
            if 'bus' not in k:
                if k == 'sidewalk':
                    objtype = Modeler.SIDEWALK
                elif k == 'wall':
                    objtype = Modeler.WALL
                else:
                    objtype = Modeler.NULL
                obj = self.__load(k, type=objtype)
                obj.name = 'CSUR_' + k
                self.objs[k] = obj
                obj.hide_set(True)
        # load bus stop; need to merge two models
        bus_r = self.__load('bus_road', type=Modeler.NULL, recenter=False)
        bus_s = self.__load('bus_side', type=Modeler.SIDEWALK, recenter=False)
        obj = make_mesh([bus_r, bus_s])
        obj.name = 'CSUR_bus_stop'
        align(obj.data)
        self.objs['bus_stop'] = obj
        obj.hide_set(True)
            

   
    def __load(self, name, type=NULL, recenter=True):
        path = os.path.join(self.config['PATH']['workdir'],
                            self.config['PATH']['units'],
                            self.config['UNITS'][name])
        bpy.ops.import_scene.fbx(filepath=path)
        obj = bpy.context.selected_objects[0]
        obj.scale = Vector([1, 1, 1])
        obj.location = Vector([0, 0, 0])
        if recenter:
            align(obj.data)
        link_image(obj, self.textures['d'][type])
        return obj

    def __build_units(self, start, end, xs_start, xs_end, mirror=False):
        deselect()
        units = [x or y for x, y in zip(start, end)]
        lb = self.lane_border
        p = 0
        objs_created = []
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.LANE:
                lane_added = 0
                x_right = [xs_start[p] + LANEWIDTH / 2, xs_end[p] + LANEWIDTH / 2]
                if units[p - 1] == Segment.CHANNEL:
                    obj = self.objs['lane_f']
                    x_left = [xs_start[p] + lb, xs_end[p] + lb]
                else:
                    obj = self.objs['lane_l']
                    x_left = [xs_start[p] - lb, xs_end[p] - lb]
                objs_created.append(place_unit(obj, x_left, x_right))
                x_left = x_right.copy()
                lane_added += 1
                for _ in range(nblocks - 1):
                    for i, xs in enumerate([xs_start, xs_end]):
                            x_right[i] = min(xs[p + lane_added] + 0.5 * LANEWIDTH,
                                            xs[p + lane_added + 1] - 0.5 * LANEWIDTH)
                        
                    obj = self.objs['lane_c']
                    uvflag = x_left[1] - x_left[0] != x_right[1] - x_right[0]
                    objs_created.append(place_unit(obj, x_left, x_right, preserve_uv=uvflag))
                    x_left = x_right.copy()
                    lane_added += 1
                if units[p + nblocks] == Segment.CHANNEL:
                    obj = self.objs['lane_f']
                    x_right = [x_left[0] + LANEWIDTH / 2 - lb, x_left[1] + LANEWIDTH /2 - lb]
                else:
                    obj = self.objs['lane_r']
                    x_right = [x_left[0] + LANEWIDTH / 2 + lb, x_left[1] + LANEWIDTH /2 + lb]
                objs_created.append(place_unit(obj, x_left, x_right))
            elif units[p] == Segment.CHANNEL:
                x0 = [xs_start[p] - lb, xs_end[p] - lb]
                x2 = [xs_start[p+nblocks] + lb, xs_end[p+nblocks] + lb]
                x1 = [(x0[0] + x2[0]) / 2, (x0[1] + x2[1]) / 2]
                obj = self.objs['channel_l']
                objs_created.append(place_unit(obj, x0, x1))
                obj = self.objs['channel_r']
                objs_created.append(place_unit(obj, x1, x2))
            elif units[p] == Segment.MEDIAN:
                if xs_start[p] == 0:
                    obj = self.objs['median_h']
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
                else:
                    obj = self.objs['median_f']
                    objs_created.append(place_unit(obj,
                            [xs_start[p] + lb, xs_end[p] + lb], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))   
            elif units[p] == Segment.BIKE:
                obj = self.objs['bike']
                objs_created.append(place_unit(obj, 
                                    [xs_start[p] - lb, xs_end[p] - lb], 
                                    [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]))
            elif units[p] == Segment.CURB:
                # add a wall to the left end of the road
                if p == 0:
                    obj = self.objs['wall']
                    objs_created.append(place_unit(obj, 
                                    [xs_start[p] + lb, xs_end[p] + lb], 
                                    [xs_start[p] + lb, xs_end[p] + lb]))
                    obj = self.objs['curb']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
                else:
                    obj = self.objs['curb']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            elif units[p] == Segment.SIDEWALK:
                obj = self.objs['sidewalk']
                objs_created.append(place_unit(obj, 
                                    [xs_start[p], xs_end[p]], 
                                    [xs_start[p + nblocks], xs_end[p + nblocks]]))
            p += nblocks
        return objs_created

    def build(self, seg):
        obj = self.__build_units(seg.start[seg.first_lane:],
                        seg.end[seg.first_lane:],
                        seg.x_start[seg.first_lane:], 
                        seg.x_end[seg.first_lane:])
        obj = make_mesh(obj)
        # If the segment is two-way road then mirror it
        # mirror of one side of the road is the reverse
        if isinstance(seg, csur.TwoWay):
            obj_r = self.__build_units(seg.end[seg.first_lane:],
                        seg.start[seg.first_lane:],
                        seg.x_end[seg.first_lane:], 
                        seg.x_start[seg.first_lane:])
            obj_r = make_mesh(obj_r)
            obj_r.rotation_euler[0] = 0
            obj_r.rotation_euler[1] = 0
            obj_r.rotation_euler[2] = 3.1415926536
            obj = make_mesh([obj, obj_r])
        obj.name = str(seg)
        # reset origin
        reset_origin(obj)
        return obj
                    