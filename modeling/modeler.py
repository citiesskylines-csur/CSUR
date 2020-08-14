import configparser, os, importlib
import bpy
from mathutils import Vector
from modeling.blender_utils import *
from core import csur
from core.csur import Segment, CSURFactory

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
    NODE = 2
    ELEVATED = 3
    BRIDGE = 4
    TUNNEL = 5
    SLOPE = 6


    def __init__(self, config_file, bridge=False, tunnel=True, lod=False, optimize=False):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        # lane_border indicates the width that a lane unit extends at the edge
        self.lane_border = float(self.config['PARAM']['lane_border'])
        self.deck_margin = float(self.config['PARAM']['deck_margin'])
        self.beam_margin = float(self.config['PARAM']['beam_margin'])
        self.median_margin = float(self.config['PARAM']['median_margin'])
        # load textures
        self.texpath = os.path.join(os.path.dirname(os.path.abspath(config_file)), self.config['PATH']['tex'])
        self.bridge = bridge
        self.tunnel = tunnel
        self.lod = lod
        self.optimize = optimize
        # road_d is the Default texture map
        self.textures = None
        if not self.optimize:
            self.textures = {'d':{}}
            self.textures['d'][Modeler.NULL] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['road_d']))
            self.textures['d'][Modeler.NODE] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['node_d']))
            self.textures['d']['BRT'] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['brt_d']))
            self.textures['d']['SB'] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['sound_barrier_d']))

            self.textures['d'][Modeler.SIDEWALK] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['sidewalk_d']))
            self.textures['d'][Modeler.ELEVATED] = bpy.data.images.load(
                                    filepath=os.path.join(self.texpath, self.config['TEX']['elevated_d']))
            if self.bridge:
                self.textures['d'][Modeler.BRIDGE] = bpy.data.images.load(
                                        filepath=os.path.join(self.texpath, self.config['TEX']['bridge_d']))
            if self.tunnel:
                self.textures['d'][Modeler.TUNNEL] = bpy.data.images.load(
                                        filepath=os.path.join(self.texpath, self.config['TEX']['tunnel_d']))

        #load models:
        self.objs = {'LANE': {}, 'GROUND': {}, 'NODE': {}, 'ELEVATED': {}, 'BRIDGE': {}, 'TUNNEL': {}, 'SLOPE': {}, 'SPECIAL': {}}
        #load lanes
        for k, v in self.config['LANE'].items():
            obj = self.__load(v, objectname='CSUR_' + k)
            self.objs['LANE'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['ELEVATED'].items():
            if k == 'sound_barrier':
                obj = self.__load(v, objectname='CSUR_elv_' + k, type='SB')
            else:
                obj = self.__load(v, objectname='CSUR_elv_' + k, type=Modeler.ELEVATED)
            self.objs['ELEVATED'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['GROUND'].items():
            if k in ['sidewalk', 'wall', 'bus_side', 'bus_side_nobike']:
                obj = self.__load(v, objectname='CSUR_gnd_' + k, type=Modeler.SIDEWALK, recenter=('bus_side' not in k))
            elif k == 'brt_station':
                obj = self.__load(v, objectname='CSUR_gnd_' + k, type='BRT', recenter=False)
            else:
                obj = self.__load(v, objectname='CSUR_gnd_' + k)

            self.objs['GROUND'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['NODE'].items():
            obj = self.__load(v, objectname='CSUR_node_' + k, type=Modeler.NODE)
            self.objs['NODE'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['SPECIAL'].items():
            obj = self.__load(v, objectname='CSUR_special_' + k)
            self.objs['SPECIAL'][k] = obj
            obj.hide_set(True)

        if self.bridge:
            for k, v in self.config['BRIDGE'].items():
                obj = self.__load(v, objectname='CSUR_bdg_' + k, type=Modeler.BRIDGE)
                self.objs['BRIDGE'][k] = obj
                obj.hide_set(True)
        if self.tunnel:
            for k, v in self.config['TUNNEL'].items():
                obj = self.__load(v, objectname='CSUR_tun_' + k, type=Modeler.TUNNEL)
                self.objs['TUNNEL'][k] = obj
                obj.hide_set(True)
            for k, v in self.config['SLOPE'].items():
                obj = self.__load(v, objectname='CSUR_slp_' + k, type=Modeler.TUNNEL, recenter=False)
                self.objs['SLOPE'][k] = obj
                obj.hide_set(True)

        # optimized mode does not use material
        if self.optimize:
            wipe_materials()

    def set_interp_type(self, interp):
        blender_utils.INTERP_TYPE = interp

    def check_mode(self, mode):
        if mode[0] == 'b' and not self.bridge:
            raise Exception("Bridge mode not loaded!")
        if mode[0] in ['t', 's'] and not self.tunnel:
            raise Exception("Tunnel mode not loaded!")

    def __load(self, filename, objectname=None, type=NULL, recenter=True):
        haslod = False
        if self.lod:
            lod_filename = ''.join(filename.split('.')[:-1] + ['_lod.'] + [filename.split('.')[-1]])
            if os.path.exists(os.path.join(self.config['PATH']['model'], lod_filename)):
                filename = lod_filename
                haslod = True
        if objectname:
            if haslod:
                if objectname + '_lod' in bpy.data.objects:
                    return bpy.data.objects[objectname + '_lod']
                objectname += '_lod'
            elif objectname in bpy.data.objects:
                return bpy.data.objects[objectname]
        path = os.path.join(self.config['PATH']['model'],
                            filename)
        bpy.ops.import_scene.fbx(filepath=path)
        obj = bpy.context.selected_objects[0]
        obj.animation_data_clear()
        obj.scale = Vector([1, 1, 1])
        obj.location = Vector([0, 0, 0])
        obj.rotation_euler = Vector([0, 0, 0])
        if recenter:
            align(obj.data)
        if self.textures:
            link_image(obj, self.textures['d'][type])
        clean_uv(obj)
        #mirror_uv(obj)
        if objectname:
            obj.name = objectname
        return obj

    def save(self, obj, path):
        deselect()
        obj.select_set(True)
        if not os.path.exists(path):
            bpy.ops.export_scene.fbx(filepath=path, 
                    axis_forward='Z', axis_up='Y', use_selection=True, bake_space_transform=True,
                    mesh_smooth_type='OFF')
        else:
            print("Warning: file %s already exists!" % path)
        if self.optimize:
            delete(obj)

    def cleanup(self):
        cleanup_scene()

    def __make_lanes(self, units, xs_start, xs_end, busstop=None, divide_line=False, central_channel=False, flip_texture=False):
        # game's texture is opposite to Blender preview
        #if self.optimize:
        #    flip_texture = not flip_texture
        deselect()
        xs_start = xs_start.copy()
        xs_end = xs_end.copy()
        units = units.copy()
        lb = self.lane_border
        p = 0
        objs_created = []
        if divide_line and units[0] == Segment.LANE:
            # determine the type of divide_line
            nlanes = sum(x == Segment.LANE for x in units)
            if Segment.BIKE not in units and Segment.SIDEWALK in units:
                # gc mode
                key = 'divide_line' if nlanes >= 3 else 'divide_line_single'
            else:
                key = 'divide_line' if nlanes >= 2 else 'divide_line_single'
            obj = self.objs['LANE'][key]
            objs_created.append(place_unit(obj, [xs_start[0]-lb, xs_end[0]-lb], [xs_start[0]+lb, xs_end[0]+lb]))
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.LANE:
                centered_trans_offset = 0
                if Segment.CHANNEL in units and (xs_start[p] - xs_end[p]) * (xs_start[p + nblocks] - xs_end[p + nblocks]) < 0:
                    centered_trans_offset = (xs_start[p + nblocks] - xs_end[p + nblocks])
                    for i in range(nblocks - 1):
                        if xs_start[p + i] == xs_start[p + i + 1] or xs_end[p + i] == xs_end[p + i + 1]:
                            xs_start.pop(p + i + 1)
                            xs_end.pop(p + i + 1)
                            units.pop(p + i + 1)      
                            break
                    nblocks -= 1                
                lane_added = 0
                x_right = [xs_start[p] + LANEWIDTH / 2, xs_end[p] + LANEWIDTH / 2]
                if p == 0 and not central_channel:
                    obj = self.objs['LANE']['lane_f']
                    x_left = [xs_start[p] + lb, xs_end[p] + lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
                elif central_channel or units[p - 1] == Segment.CHANNEL:
                    if Segment.LANE in units[:p] or (not central_channel and (xs_start[p-1] == xs_start[p] or xs_end[p-1] == xs_end[p])):
                        obj = self.objs['LANE']['lane_h']
                    else:
                        obj = self.objs['LANE']['lane_r']
                    if xs_end[p - 1] == xs_end[p]:
                        obj = make_mirror(obj, axis=1, copy=True)
                    else:
                        obj = duplicate(obj)
                    obj_temp = place_unit(obj, [0, 0], [get_dims(obj.data)[0] - lb, get_dims(obj.data)[0] - lb], preserve_uv=1, copy=False)
                    make_mirror(obj_temp, copy=False)
                    x_left = [xs_start[p], xs_end[p]]
                    objs_created.append(place_unit(obj_temp, x_left, x_right, copy=False))
                elif units[p - 1] == Segment.WEAVE:
                    obj = self.objs['LANE']['lane_f']
                    x_left = [xs_start[p] - LANEWIDTH / 4 + lb, xs_end[p] - LANEWIDTH / 4 + lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
                else: 
                    if Segment.SIDEWALK in units[:p] and busstop == 'double':
                        obj = self.objs['LANE']['lane_f']
                        x_left = [xs_start[p], xs_end[p]]
                    else:
                        obj = self.objs['LANE']['lane_l']
                        x_left = [xs_start[p] - lb, xs_end[p] - lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
                if centered_trans_offset:
                    x_left = x_right.copy()
                    obj = self.objs['LANE']['lane_f']
                    x_right = [max(x_left), max(x_left)]
                    objs_created.append(place_unit(obj, x_left, x_right))
                    if nblocks == 1:
                        x_right[xs_start[p] - xs_end[p] >= 0] += LANEWIDTH / 2
                x_left = x_right.copy()
                lane_added += 1
                for j in range(nblocks - 1):
                    for i, xs in enumerate([xs_start, xs_end]):
                        # explanation of 'magic':
                        # the lane center model spans from (x - 0.5) to (x + 0.5) LW
                        # however there could be an EMPTY in the next unit so 
                        # this lane center should shrink
                        # the rule of thumb is that it NEVER INTRUDES THE REGION OF NEXT MODEL
                        # which could be defined by either (x + 0.5) or (next(x) - 0.5) LW
                        x_right[i] = min(xs[p + lane_added] + 0.5 * LANEWIDTH,
                                        xs[p + lane_added + 1] - 0.5 * LANEWIDTH)
                        # besides, the right boundary should not cross the left boundary
                        x_right[i] = max(x_right[i], x_left[i])
                    uvflag = int(x_left[1] - x_left[0] != x_right[1] - x_right[0])
                    if j == (nblocks - 2) and centered_trans_offset:
                        x_temp = [max(x_right), max(x_right)]
                        obj = self.objs['LANE']['lane_c']
                        objs_created.append(place_unit(obj, x_left, x_temp, preserve_uv=uvflag))
                        x_left = x_temp.copy()
                    else:
                       
                        if x_left[0] != x_left[1] and x_right[0] != x_right[1] and x_left[1] - x_left[0] != x_right[1] - x_right[0]:
                            obj = duplicate(self.objs['LANE']['lane_f'])
                            obj = place_unit(obj, [0, 0], 
                                                  [max(x_right[0] - x_left[0], EPS/2), max(x_right[1] - x_left[1], EPS/2)],
                                                   preserve_uv=0, interpolation='linear', copy=False)
                            uvflag = 0
                            scale_mode = 2
                        else:
                            obj = duplicate(self.objs['LANE']['lane_c'])
                            scale_mode = 0
                        objs_created.append(place_unit(obj, x_left, x_right, preserve_uv=uvflag, copy=False, scale_mode=scale_mode))    
                        x_left = x_right.copy()
                    lane_added += 1
                if centered_trans_offset:
                    if x_left[0] == x_left[1]:
                        print('left:', x_left, centered_trans_offset)
                        x_temp = x_left.copy()
                        x_left[0] += max(centered_trans_offset, 0)
                        x_left[1] += -min(centered_trans_offset, 0)
                    else:
                        x_temp = [min(x_left), min(x_left)]
                    objs_created.append(place_unit(self.objs['LANE']['lane_f'], x_temp, x_left))
                if units[p + nblocks] == Segment.CHANNEL:
                    obj = self.objs['LANE']['lane_h' \
                                if xs_start[p+nblocks] == xs_start[p+nblocks+1] \
                                    or xs_end[p+nblocks] == xs_end[p+nblocks+1] else 'lane_r']
                    if xs_end[p + nblocks] == xs_end[p + nblocks + 1]:
                        obj = make_mirror(obj, axis=1, copy=True)
                    else:
                        obj = duplicate(obj)
                    obj_temp = place_unit(obj, [0, 0], [get_dims(obj.data)[0] - lb, get_dims(obj.data)[0] - lb],
                                preserve_uv=1, copy=False)
                    x_right = [x_left[0] + LANEWIDTH / 2, x_left[1] + LANEWIDTH /2]
                    objs_created.append(place_unit(obj_temp, x_left, x_right, copy=False))
                elif units[p + nblocks] == Segment.WEAVE:
                    obj = self.objs['LANE']['lane_f']
                    x_right = [x_left[0] + 3 * LANEWIDTH / 4 - lb, x_left[1] + 3 * LANEWIDTH /4 - lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
                else:
                    if not Segment.LANE in units[p + nblocks:] and Segment.SIDEWALK in units[p + nblocks:] and busstop in ['single', 'double']:
                        obj = self.objs['LANE']['lane_f']
                        x_right = [x_left[0] + LANEWIDTH / 2, x_left[1] + LANEWIDTH /2]
                    else:
                        obj = self.objs['LANE']['lane_r']
                        x_right = [x_left[0] + LANEWIDTH / 2 + lb, x_left[1] + LANEWIDTH /2 + lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
            elif units[p] == Segment.CHANNEL:
                x0 = [xs_start[p], xs_end[p]]
                x2 = [xs_start[p+nblocks], xs_end[p+nblocks]]
                x1 = [(x0[0] + x2[0]) / 2, (x0[1] + x2[1]) / 2]
                if p == 0:
                    # the mesh of central channel should be flipped
                    # to ensure normal game behavior
                    # also uvflag should be flipped
                    if divide_line:
                        obj = self.objs['LANE']['channel_c']
                        w = get_dims(obj.data)[0]
                        obj_temp = place_unit(obj, [0, 0], [2*w, 2*w], 
                                                preserve_uv=1, interpolation='linear')
                        align(obj_temp.data)
                        #if not flip_texture:
                        if flip_texture:
                            obj_temp = make_mirror(obj_temp, copy=False)
                            uvflag = -1
                        else:
                            uvflag = 1
                        #obj_temp = make_mirror(obj, copy=False)
                        #print(x0, x2)
                        if x0[0] == x2[0]:
                            if x0[0] + x2[0] == x0[1] + x2[1]:
                                obj_temp = place_unit(obj_temp, [w, 0], [w + EPS/2, 2 * w], 
                                                preserve_uv=uvflag, interpolation='linear', copy=False)
                            else:
                                obj_temp = place_unit(obj_temp, [0, 0], [EPS/2, 2 * w], 
                                                    preserve_uv=uvflag, interpolation='linear', copy=False)
                        elif x0[1] == x2[1]:
                            #obj_temp = invert(obj_temp, axis=2, copy=False)
                            obj_temp = make_mirror(obj_temp, axis=1, copy=False)
                            uvflag *= -1
                            mirror_uv(obj_temp, axis=1)
                            if x0[0] + x2[0] == x0[1] + x2[1]:
                                obj_temp = place_unit(obj_temp, [0, w], [2 * w, w + EPS/2], 
                                                    preserve_uv=-uvflag, interpolation='linear', copy=False)
                            else:
                                obj_temp = place_unit(obj_temp, [0, 0], [2 * w, EPS/2], 
                                                    preserve_uv=-uvflag, interpolation='linear', copy=False)   
                        objs_created.append(place_unit(obj_temp, [2*x0[0]-x2[0],2*x0[1]-x2[1]], x2, scale_mode=2, copy=False))
                else:
                    obj = self.objs['LANE']['channel']
                    if flip_texture:
                        obj_temp = make_mirror(obj)
                        uvflag = -1
                    else:
                        obj_temp = duplicate(obj)
                        uvflag = 1
                    # default channel model forks forward
                    if x0[0] == x2[0]:
                        obj_temp = place_unit(obj_temp, [0,0], [EPS/2,get_dims(obj.data)[0]], 
                                                preserve_uv=uvflag, interpolation='linear', copy=False)
                    elif x0[1] == x2[1]:
                        #obj_temp = invert(obj_temp, axis=2, copy=False)
                        obj_temp = make_mirror(obj_temp, axis=1, copy=False)
                        uvflag *= -1
                        mirror_uv(obj_temp, axis=1)
                        obj_temp = place_unit(obj_temp, [0,0], [get_dims(obj.data)[0], EPS/2], 
                                                preserve_uv=-uvflag, interpolation='linear', copy=False)
                    objs_created.append(place_unit(obj_temp, x1, x2, scale_mode=2))
                    obj_temp = make_mirror(obj_temp, copy=False)
                    objs_created.append(place_unit(obj_temp, x0, x1, scale_mode=2, copy=False))
            elif units[p] == Segment.SHOULDER or units[p] == Segment.HALFSHOULDER:
                obj = self.objs['LANE']['lane_f']
                if p >= 1 and units[p - 1] == Segment.BARRIER:
                    x_left = [xs_start[p] - lb, xs_end[p] - lb]
                else:
                    x_left = [xs_start[p] + lb, xs_end[p] + lb]
                if units[p + nblocks] != Segment.LANE:
                    x_right = [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]
                else:
                    x_right = [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]
                objs_created.append(place_unit(obj, x_left, x_right))
            elif units[p] == Segment.WEAVE:
                obj = self.objs['LANE']['weave'] 
                objs_created.append(place_unit(obj,
                        [xs_start[p] + LANEWIDTH / 4 - lb, xs_end[p] + LANEWIDTH / 4 - lb], 
                        [xs_start[p + nblocks] - LANEWIDTH / 4 + lb, xs_end[p + nblocks] - LANEWIDTH / 4 + lb]))   
            p += nblocks
        return objs_created

    def __make_ground(self, units, xs_start, xs_end, busstop=None):
        deselect()
        lb = self.lane_border
        p = 0
        lanes_extra = []
        struc = []
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN:
                if p == 0:
                    obj = self.objs['GROUND']['median_h']
                    lanes_extra.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            ))
                else:
                    if busstop in ['single', 'double'] and Segment.SIDEWALK in units[p+nblocks:] and Segment.MEDIAN not in units[p+nblocks:]:
                        nblocks = 4
                        obj = self.objs['GROUND']['bus_road']
                        lanes_extra.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            preserve_obj=1))
                        obj = self.objs['GROUND']['bus_side']
                        struc.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            preserve_obj=1))   
                    else:
                        if busstop == 'brt' and nblocks == 2:
                            obj = self.objs['GROUND']['brt_median']
                        else:
                            obj = self.objs['GROUND']['median_f']
                        lanes_extra.append(place_unit(obj,
                                [xs_start[p] + lb, xs_end[p] + lb], 
                                [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))   
            elif units[p] == Segment.BIKE:
                if units[p - 1] == Segment.WEAVE:
                    x_left = [xs_start[p] + lb - LANEWIDTH / 4, xs_end[p] + lb - LANEWIDTH / 4]
                else:
                    x_left = [xs_start[p] - lb, xs_end[p] - lb]
                obj = self.objs['GROUND']['bike']
                lanes_extra.append(place_unit(obj, 
                                    x_left, 
                                    [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]))
            elif units[p] == Segment.CURB:
                # add a wall to the left end of the road
                # p<=1 does not guarantee being a left curb. eg., in 2DCc units[1] is a right curb
                if p <= 1 and units[0] != Segment.LANE:
                    if p == 0:
                        obj = self.objs['GROUND']['wall']
                        struc.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p], xs_end[p]]))
                    obj = self.objs['GROUND']['curb']
                    obj_temp = make_mirror(obj)       
                    lanes_extra.append(place_unit(obj_temp, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb], copy=False))
                elif busstop in ['single', 'double'] and Segment.BIKE not in units:
                    obj = self.objs['GROUND']['bus_road_nobike']
                    lanes_extra.append(place_unit(obj,
                        [xs_start[p], xs_end[p]], 
                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                        preserve_obj=1))
                    obj = self.objs['GROUND']['bus_side_nobike']
                    struc.append(place_unit(obj,
                        [xs_start[p], xs_end[p]], 
                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                        preserve_obj=1))
                    nblocks = 2
                else:
                    obj = self.objs['GROUND']['curb']            
                    lanes_extra.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
                    if p == len(units) - 1:
                        obj = make_mirror(self.objs['GROUND']['wall'])
                        struc.append(place_unit(obj, 
                                        [xs_start[p + 1], xs_end[p + 1]], 
                                        [xs_start[p + 1], xs_end[p + 1]], copy=False))
            elif units[p] == Segment.SIDEWALK:
                # left busstop
                if p == 0 and busstop == 'double':
                    # ground normal
                    if units[p + 2] == Segment.BIKE:
                        nblocks = 4
                        bus_road = make_mirror(self.objs['GROUND']['bus_road'], realign=False)
                        bus_side = make_mirror(self.objs['GROUND']['bus_side'], realign=False)
                    else:
                        nblocks = 2
                        bus_road = make_mirror(self.objs['GROUND']['bus_road_nobike'], realign=False)
                        bus_side = make_mirror(self.objs['GROUND']['bus_side_nobike'], realign=False)
                    bus_road = place_unit(bus_road,
                        [xs_start[p + nblocks], xs_end[p + nblocks]], 
                        [xs_start[p + nblocks], xs_end[p + nblocks]],
                        preserve_obj=1, copy=False)
                    bus_side = place_unit(bus_side,
                        [xs_start[p + nblocks], xs_end[p + nblocks]], 
                        [xs_start[p + nblocks], xs_end[p + nblocks]],
                        preserve_obj=1, copy=False)
                    lanes_extra.append(bus_road)
                    struc.append(bus_side)
                else:
                    obj = self.objs['GROUND']['sidewalk']
                    obj = make_mirror(obj) if p == 0 else duplicate(obj)
                    struc.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]], copy=False))
            p += nblocks
        return lanes_extra, struc

    def __make_bridge(self, units, xs_start, xs_end, divide_line=False):
        lb = self.lane_border
        bw = get_dims(self.objs['BRIDGE']['beam'].data)[0]
        objs_created = []
        # make beams
        w_lanes = max(xs_end[-2] - xs_end[1], xs_start[-2] - xs_start[1])
        n_beams = int(w_lanes // bw) + 1
        scale = w_lanes / (bw * n_beams)
        beams = []
        if units[0] == Segment.BARRIER:
            xs_0 = [xs_start[1] - lb, xs_end[1] - lb]
        else:
            xs_0 = [xs_start[0] , xs_end[0]]
        xs = [xs_0[0], xs_0[0]]
        for i in range(n_beams):
            beams.append(place_unit(self.objs['BRIDGE']['beam'], xs,
                                [xs[0] + bw, xs[1] + bw]
                            ))
            xs[0], xs[1] = xs[0] + bw, xs[1] + bw
        beam_obj = make_mesh(beams)
        beam_obj.scale[0] = scale
        transform_apply(beam_obj, scale=True)
        align(beam_obj.data, axis=0)
        objs_created.append(place_unit(beam_obj, xs_0, [xs_start[-2], xs_end[-2]], copy=False, scale_mode=1))
        # make bridge deck
        obj = self.objs['BRIDGE']['deck_h'] if units[0] != Segment.BARRIER else self.objs['BRIDGE']['deck_f']
        obj_scaled = duplicate(obj)
        obj_scaled.scale[0] = scale
        transform_apply(obj_scaled, scale=True)
        align(obj_scaled.data, axis=0)
        idx = int(units[0] == Segment.BARRIER)
        objs_created.append(place_unit(obj_scaled, 
                            [xs_start[idx] - lb * idx, xs_end[idx] - lb * idx],
                            [xs_start[-2], xs_end[-2]],
                            copy=False))
        objs_created.append(place_unit(self.objs['ELEVATED']['joint'], 
                            [xs_start[idx] - lb * idx, xs_end[idx] - lb * idx],
                            [xs_start[-2], xs_end[-2]],
                            preserve_uv = 2
                            ))
        # add median and barrier
        lb = self.lane_border
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.CHANNEL:
                if xs_start[p] == 0 and divide_line:
                    obj = self.objs['BRIDGE']['median']
                    objs_created.append(place_unit(obj,
                            [-xs_start[p + nblocks], -xs_start[p + nblocks]], 
                            [xs_start[p + nblocks], xs_end[p + nblocks]],
                            ))
            elif units[p] == Segment.BARRIER:
                obj = self.objs['BRIDGE']['barrier']
                width = get_dims(obj.data)[0]
                if p == 0:      
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p + nblocks] - lb - width, xs_end[p + nblocks] - lb - width], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
                else:
                    # Loaded model is left barrier
                    obj = make_mirror(obj)            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p] + width, xs_end[p] + width], copy=False))
            elif units[p] == Segment.SIDEWALK:
                obj = self.objs['BRIDGE']['sidewalk']
                if xs_start[p] < 0 and xs_end[p] < 0:
                    obj = make_mirror(obj)
                else:
                    obj = duplicate(obj)
                objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]], copy=False))
            p += nblocks
        return objs_created

    def __make_tunnel(self, units, xs_start, xs_end, ncall=[0]):
        objs_created = []
        lb = self.lane_border
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN:
                if p == 0:
                    obj = self.objs['TUNNEL']['median']
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            ))
                else:
                    raise ValueError('Tunnel should not have non-central median')
            elif units[p] == Segment.BARRIER:
                obj = self.objs['TUNNEL']['barrier']        
                if p == 0:
                    # Loaded model is right barrier
                    obj = make_mirror(obj)          
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],  copy=False))
                else:         
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            elif units[p] == Segment.CHANNEL:
                dx = float(self.config['PARAM']['tunnel_bevel'])
                obj = duplicate(self.objs['TUNNEL']['gore'])
                if xs_start[p] == xs_start[p + nblocks]:
                    x0 = [xs_end[p] + lb - dx] * 2
                    x1 = [xs_end[p+nblocks] - lb + dx] * 2
                    objs_created.append(place_unit(obj, x0, x1, copy=False))
                elif xs_end[p] == xs_end[p + nblocks]:
                    x0 = [xs_start[p] + lb - dx] * 2
                    x1 = [xs_start[p+nblocks] - lb + dx] * 2
                    obj = make_mirror(obj, axis=1, realign=False, copy=False)
                    objs_created.append(place_unit(obj, x0, x1, copy=False))           
            
            p += nblocks
        p = 0
        while units[p] in [Segment.MEDIAN, Segment.BARRIER]:
            p += 1
        obj = self.objs['TUNNEL']['roof']
        objs_created.append(place_unit(obj, [xs_start[p] - lb, xs_end[p] - lb], [xs_start[-2] + lb, xs_end[-2] + lb]))
        return objs_created

    def __make_slope(self, units, xs_start, xs_end, reverse=True):
        deselect()
        objs_created = []
        lb = self.lane_border
        p = 0
        seam = float(self.config['PARAM']['slope_median_seam']) if not self.lod else 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.CHANNEL:
                if p == 0:
                    obj = duplicate(self.objs['SLOPE']['median'])
                    if reverse:
                        obj = make_mirror(obj, axis=1, copy=False)
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb + seam, xs_end[p + nblocks] - lb + seam],copy=False
                            ))
            elif units[p] == Segment.BARRIER:
                obj = duplicate(self.objs['SLOPE']['barrier'])
                if reverse:
                    obj = make_mirror(obj, axis=1, copy=False)
                if p == 0:
                    # Loaded model is right barrier
                    obj = make_mirror(obj, copy=False)          
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] - seam, xs_end[p] - seam], 
                                        [xs_start[p + nblocks] - lb + seam, xs_end[p + nblocks] - lb + seam],  copy=False))
                else:         
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb - seam, xs_end[p] + lb - seam], 
                                        [xs_start[p + nblocks] + seam, xs_end[p + nblocks] + seam], copy=False))     
            p += nblocks
        p = 0
        while units[p] in [Segment.CHANNEL, Segment.BARRIER]:
            p += 1
        obj = duplicate(self.objs['SLOPE']['roof'])
        if reverse:
            obj = make_mirror(obj, axis=1, copy=False)
        objs_created.append(place_unit(obj, 
                            [xs_start[p] - lb, xs_end[p] - lb], [xs_start[-2] + lb, xs_end[-2] + lb], copy=False))
        w_lanes = max(xs_end[-2] - xs_end[units.index(Segment.LANE)], xs_start[-2] - xs_start[units.index(Segment.LANE)])
        obj = duplicate(self.objs['SLOPE']['arch'] if w_lanes > 3 * LANEWIDTH else self.objs['SLOPE']['arch2'])
        if reverse:
            obj = make_mirror(obj, axis=1, copy=False)
        objs_created.append(place_unit(obj, 
                            [xs_start[p] - lb, xs_end[p] - lb], [xs_start[-2] + lb, xs_end[-2] + lb], scale_mode=1, copy=False))
        return objs_created

    def __make_elevated(self, units, xs_start, xs_end, joint=True):
        dm, bm, mm = self.deck_margin, self.beam_margin, self.median_margin
        lb = self.lane_border
        bs = get_dims(self.objs['ELEVATED']['beam_sep'].data)[0]
        bw = get_dims(self.objs['ELEVATED']['beam'].data)[0]
        objs_created = []
        if not self.lod:
            # make beams
            w_lanes = max(xs_end[-2] - xs_end[1], xs_start[-2] - xs_start[1])
            w_beam_max = bs + bw
            n_beams = int(w_lanes // (w_beam_max)) + 1
            scale = w_lanes / (w_beam_max * n_beams)
            beams = []
            if units[0] == Segment.MEDIAN:
                xs_0 = [xs_start[1] - mm, xs_end[1] - mm]
            elif units[0] == Segment.BARRIER:
                xs_0 = [xs_start[0] + bm, xs_end[0] + bm]
            elif units[0] == Segment.LANE or units[0] == Segment.CHANNEL:
                xs_0 = [xs_start[0], xs_end[0]]
            else:
                raise ValueError('Cannot make deck model: not an elevated segment! %s', units)
            xs = [xs_0[0], xs_0[0]]
            beams.append(place_unit(self.objs['ELEVATED']['beam_sep'], xs,
                                    [xs_start[1] + bs / 2 - lb, xs_start[1] + bs / 2 - lb]
                                ))
            xs[0], xs[1] = xs_start[1] + bs / 2 - lb, xs_start[1] + bs / 2 - lb
            for i in range(n_beams):
                beams.append(place_unit(self.objs['ELEVATED']['beam'], xs,
                                    [xs[0] + bw, xs[1] + bw]
                                ))
                xs[0], xs[1] = xs[0] + bw, xs[1] + bw
                if i < n_beams - 1:
                    beams.append(place_unit(self.objs['ELEVATED']['beam_sep'], xs,
                                    [xs[0] + bs, xs[1] + bs]
                                ))
                    xs[0], xs[1] = xs[0] + bs, xs[1] + bs
            beams.append(place_unit(self.objs['ELEVATED']['beam_sep'], xs,
                                    [xs[0] + bs / 2 + bm + lb, xs[1] + bs / 2 + bm + lb]
                                ))
            beam_obj = make_mesh(beams)
            beam_obj.scale[0] = scale
            transform_apply(beam_obj, scale=True)
            align(beam_obj.data, axis=0)
            objs_created.append(place_unit(beam_obj, xs_0, [xs_start[-1] - bm, xs_end[-1] - bm], copy=False, scale_mode=1))
            # make bridge deck
            obj = self.objs['ELEVATED']['deck_h'] if units[0] != Segment.BARRIER else self.objs['ELEVATED']['deck_f']
            obj_scaled = duplicate(obj)
            obj_scaled.scale[0] = scale
            transform_apply(obj_scaled, scale=True)
            align(obj_scaled.data, axis=0)
            if units[0] != Segment.BARRIER:
                objs_created.append(place_unit(obj_scaled, 
                                    [xs_start[0], xs_end[0]],
                                    [xs_start[-1] - dm, xs_end[-1] - dm],
                                    copy=False))
                if joint:
                    objs_created.append(place_unit(self.objs['ELEVATED']['joint'], 
                                        [xs_start[0], xs_end[0]],
                                        [xs_start[-1] - dm, xs_end[-1] - dm],
                                        preserve_uv = -2
                                        ))
            else:
                objs_created.append(place_unit(obj_scaled,
                                    [xs_start[0] + dm, xs_end[0] + dm],
                                    [xs_start[-1] - dm, xs_end[-1] - dm],
                                    copy=False))
                if joint:
                    objs_created.append(place_unit(self.objs['ELEVATED']['joint'], 
                                        [xs_start[0] + dm, xs_end[0] + dm],
                                        [xs_start[-1] - dm, xs_end[-1] - dm],
                                        preserve_uv = -2
                                        ))
        # add median and barrier
        lb = self.lane_border
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN:
                if p == 0:
                    obj = self.objs['ELEVATED']['median_h']
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            ))
                else:
                    raise ValueError('Elevated road should not have non-central median')
            elif units[p] == Segment.BARRIER:
                obj = self.objs['ELEVATED']['barrier']        
                if p == 0:
                    # Loaded model is right barrier
                    obj = make_mirror(obj)          
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],  copy=False))
                else:
                    # Loaded model is left barrier          
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            elif units[p] == Segment.CHANNEL:
                if not self.lod:
                    obj = duplicate(self.objs['ELEVATED']['gore'])
                    if xs_start[p] == xs_start[p + nblocks]:
                        x0 = [xs_end[p] + lb] * 2
                        x1 = [xs_end[p+nblocks] - lb] * 2
                        objs_created.append(place_unit(obj, x0, x1, copy=False))
                    elif xs_end[p] == xs_end[p + nblocks]:
                        x0 = [xs_start[p] + lb] * 2
                        x1 = [xs_start[p+nblocks] - lb] * 2
                        obj = make_mirror(obj, axis=1, realign=False, copy=False)
                        objs_created.append(place_unit(obj, x0, x1, copy=False))              
            p += nblocks
        return objs_created

    def __check_busstop(self, seg, busstop):
        if Segment.SIDEWALK not in seg.start and busstop:
            raise ValueError("Cannot make bus stop on this segment, no sidewalk is present!")

    def __place_brt_station(self, seg, reverse=False):
        if seg.roadtype() != 'b':
            raise ValueError("Only base modules can add BRT station")
        units = [x or y for x, y in zip(seg.start, seg.end)]
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN and p > 0:
                if nblocks != 2:
                    raise ValueError("BRT station should have two units of median!")
                x = (seg.x_start[p] + seg.x_start[p + nblocks]) / 2
                brt = place_unit(self.objs['GROUND']['brt_station'], [x, x], [x, x], preserve_obj=True)
                break
            p += nblocks
        brt.location[2] = 0.15
        reset_origin(brt)
        return brt


    def __make_segment(self, seg, mode, busstop, divide_line=False, flip_texture=False):
        deselect()
        self.__check_busstop(seg, busstop)
        units = [x or y for x, y in zip(seg.start, seg.end)]
        x_start, x_end = seg.x_start, seg.x_end
        p = 0
        while units[p] == Segment.MEDIAN:
            if mode[0] in ['b', 's']:
                units[p] = Segment.CHANNEL
            p += 1
        for i in range(p, len(units)):
            if units[i] == Segment.MEDIAN and mode[0] != 'g':
                units[i] = Segment.SHOULDER
        # place traffic lanes
        lanes = self.__make_lanes(units, x_start, x_end, busstop, divide_line, flip_texture=flip_texture)
        # place ground units
        if mode[0] == 'g':
            lanes_extra, struc = self.__make_ground(units, x_start, x_end, busstop)
            lanes.extend(lanes_extra)
        # place elevated units
        elif mode[0] == 'e':
            struc = self.__make_elevated(units, x_start, x_end, joint=(type(seg) != csur.Shift))
        elif mode[0] == 'b':
            struc = self.__make_bridge(units, x_start, x_end, divide_line)
        elif mode[0] == 't':
            struc = self.__make_tunnel(units, x_start, x_end)
        elif mode[0] == 's':
            struc = self.__make_slope(units, x_start, x_end, reverse=not divide_line)
        lanes = make_mesh(lanes)
        reset_origin(lanes)
        # do not merge bridge mesh to preserve double-sided faces
        if struc:
            struc = make_mesh(struc, merge=mode[0] != 'b')        
            reset_origin(struc)
        else:
            struc = None
        return lanes, struc

    def __make_undivided(self, seg, mode, busstop):
        uleft = [x or y for x, y in zip(seg.left.start, seg.left.end)]
        uright = [x or y for x, y in zip(seg.right.start, seg.right.end)]
        for u in [uleft, uright]:
            if Segment.MEDIAN in u:
                for i in range(u.index(Segment.MEDIAN), len(u)):
                    if u[i] == Segment.MEDIAN and mode[0] != 'g':
                        u[i] = Segment.SHOULDER

        xsleft, xeleft = seg.left.x_start, seg.left.x_end
        xsright, xeright = seg.right.x_start, seg.right.x_end
        central_channel = uleft[0] == Segment.CHANNEL or uright[0] == Segment.CHANNEL
        lanes_f = self.__make_lanes(uright, xsright, xeright, busstop, divide_line=True, central_channel=central_channel)
        lanes_r = self.__make_lanes(uleft, xsleft, xeleft, None if busstop == 'single' else busstop, 
                                    central_channel=central_channel, flip_texture=True)

        units = uleft[::-1] + uright
        x_start = [-x for x in xeleft[::-1]] + xsright[1:]
        x_end = [-x for x in xsleft[::-1]] + xeright[1:]
        if mode[0] == 'g':
            lanes_f_extra, struc_f = self.__make_ground(uright, xsright, xeright, busstop)
            lanes_f.extend(lanes_f_extra)
            lanes_r_extra, struc_r = self.__make_ground(uleft, xsleft, xeleft, None if busstop == 'single' else busstop)
            lanes_r.extend(lanes_r_extra)
            struc = struc_f + struc_r
            if struc_f:
                struc_f = make_mesh(struc_f, merge=mode[0] != 'b')
                struc_r = make_mesh(struc_r, merge=mode[0] != 'b')
                reset_origin(struc_r)
                struc_r.rotation_euler[2] = 3.1415926536
                struc = [struc_f, struc_r]
            else:
                struc = None
        elif mode[0] == 'e':
            struc = self.__make_elevated(units, x_start, x_end)
        elif mode[0] == 'b':
            struc = self.__make_bridge(units, x_start, x_end)
        elif mode[0] == 't':
            struc = self.__make_tunnel(units, x_start, x_end)
        elif mode[0] == 's':
            struc = self.__make_slope(units, x_start, x_end, reverse=False)
        
        lanes_f = make_mesh(lanes_f)
        lanes_r = make_mesh(lanes_r)
        reset_origin(lanes_f)
        reset_origin(lanes_r)

        if struc:
            struc = make_mesh(struc, merge=mode[0] != 'b')
            reset_origin(struc)
        return lanes_f, lanes_r, struc
        

    def make_arrows(self, seg):
        deselect()
        if isinstance(seg, csur.TwoWay):
            arrow_f = Modeler.make_arrows(self, seg.right)
            arrow_r = Modeler.make_arrows(self, seg.left)
            arrow_r.rotation_euler[2] = 3.1415926536
            arrows = make_mesh([arrow_f, arrow_r])
        else:
            p = 0
            wmin = LANEWIDTH / 20
            units = [x or y for x, y in zip(seg.start, seg.end)]
            xs_start, xs_end = seg.x_start, seg.x_end
            arrows = []
            for i, u in enumerate(units):
                if u == Segment.LANE:
                    arrows.append(place_unit(self.objs['TUNNEL']['arrow'], 
                                            [xs_start[i], xs_end[i]], 
                                            [max(xs_start[i] + wmin, xs_start[i + 1]),
                                             max(xs_end[i] + wmin, xs_end[i + 1])], scale_mode=1))
            arrows = make_mesh(arrows)
            reset_origin(arrows)
        arrows.name = str(seg) + '_tunnel_arrows'
        return arrows
        

    def make_solidlines(self, seg, both=False):
        deselect()
        if isinstance(seg, csur.TwoWay):
            if both:
                line_l = Modeler.make_solidlines(self, seg.left)
                if line_l:
                    line_l.rotation_euler[2] = 3.1415926536
                    transform_apply(line_l, rotation=True)
                line_r = Modeler.make_solidlines(self, seg.right)
                if line_l and line_r:
                    line = make_mesh([line_l, line_r])
                else:
                    line = line_l or line_r
                return line
            else:
                return Modeler.make_solidlines(self, seg.right)
        p = 0
        # use AND instead of OR because line is only 
        # placed between two full lanes
        units = [x and y for x, y in zip(seg.start, seg.end)]
        xs_start, xs_end = seg.x_start, seg.x_end
        lines = []
        obj = self.objs['LANE']['line']
        dx = get_dims(obj.data)[0] / 2
        for i in range(1, len(units)):
            if units[i] == units[i - 1] == Segment.LANE:
                lines.append(place_unit(obj, [xs_start[i] - dx, xs_end[i] - dx], [xs_start[i] + dx, xs_end[i] + dx]))
        if len(lines) == 0:
            return None
        lines = make_mesh(lines)
        reset_origin(lines)
        lines.location[2] += 3 * EPS
        transform_apply(lines)
        lines.name = str(seg) + '_white_lines'
        return lines

    def make_soundbarrier(self, seg):
        deselect()
        if isinstance(seg, csur.TwoWay):
            sb_l = Modeler.make_soundbarrier(self, seg.left)
            sb_r = Modeler.make_soundbarrier(self, seg.right)
            sb_l.rotation_euler[2] = 3.1415926536
            sb = make_mesh([sb_l, sb_r])
            transform_apply(sb, rotation=True)
        else:
            units = [x or y for x, y in zip(seg.start, seg.end)]
            p = 0
            sb = []
            while p < len(units):
                nblocks = 1
                while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                        or units[p + nblocks] == Segment.EMPTY):
                    nblocks += 1
                if units[p] == Segment.BARRIER:
                    obj = self.objs['ELEVATED']['sound_barrier']
                    dim = get_dims(obj.data)[0]
                    if p == 0:
                        pass
                    else:
                        dx = float(self.config['PARAM']['sound_barrier_x'])
                        dz = float(self.config['PARAM']['sound_barrier_z'])
                        obj = duplicate(obj)
                        sb.append(place_unit(obj, [seg.x_start[p + nblocks] - dx - dim , seg.x_end[p + nblocks] - dx - dim],
                                             [seg.x_start[p + nblocks] - dx , seg.x_end[p + nblocks] - dx],
                                             scale_mode=1, copy=False))
                p += nblocks
            sb = make_mesh(sb)
            reset_origin(sb)
            sb.location[2] = dz
            transform_apply(sb, location=True)
        sb.name = str(seg) + "_soundbarrier"
        return sb



    def make_uturn(self, seg):
        if isinstance(seg, csur.TwoWay):
            lanes_f, struc_f = Modeler.make_uturn(self, seg.right)
            lanes_r, struc_r = Modeler.make_uturn(self, seg.left)
            lanes_r = make_mirror(lanes_r, copy=False, realign=False)
            struc_r = make_mirror(struc_r, copy=False, realign=False)
            lanes = make_mesh([lanes_f, lanes_r])
            if struc_r:
                struc = make_mesh([struc_f, struc_r])
            else:
                struc_r = None
        else:
            if seg.roadtype() != 't' or seg.x_start[-1] != seg.x_end[-1] or abs(seg.n_lanes()[0] - seg.n_lanes()[1]) > 1:
                raise ValueError("U-turn segment is only available in left +/-1 transition modules!")
            # use a temporary segment to add lanes
            # find the index where both ends are lanes
            seg_temp = seg.copy()
            p = 0
            while not (seg_temp.start[p] == seg_temp.end[p] == Segment.LANE):
                p += 1
            seg_temp.start, seg_temp.end = seg_temp.start[p:], seg_temp.end[p:]
            seg_temp.x_start, seg_temp.x_end = seg_temp.x_start[p:], seg_temp.x_end[p:]
            uturn_key = 'uturn_%dl' % (seg_temp.x_start[0] / LANEWIDTH)
            if uturn_key in self.objs['SPECIAL']:
                lanes_extra = place_unit(self.objs['SPECIAL'][uturn_key], [0,0],[0,0], preserve_obj=True)
                lanes, struc = self.__make_segment(seg_temp, 'g', None)
                # remove left border of the lane mesh
                strip(lanes, seg_temp.x_start[0], seg_temp.x_start[0] + 0.5, axis=0)
                lanes = make_mesh([lanes, lanes_extra])
            else:
                raise ValueError("Cannot make u-turn segment, need turning lane model")
        lanes.name = str(seg) + '_uturn_lanes'   
        struc.name = str(seg) + '_uturn_structure' 
        return lanes, struc




    '''
    Note on bus stop configuration:
    None: no bus stop is placed
    'single': a regular bus stop is placed on the right side of the road
    'double': a regular bus stop is placed on both sides of the road
    'brt': a BRT stop is placed on the non-central median of the road;
       the median should be two units (one lane width) wide.
    '''
    def make(self, seg, mode='g', busstop=None):
        deselect()
        busstop = busstop and busstop.lower()
        self.check_mode(mode)
        if isinstance(seg, csur.TwoWay):
            if seg.undivided:
                lanes_f, lanes_r, struc = self.__make_undivided(seg, mode, busstop)     
            else:
                lanes_f, struc_f = self.__make_segment(seg.right, mode, busstop, divide_line=True)
                lanes_r, struc_r = self.__make_segment(seg.left, mode, None if busstop == 'single' else busstop, flip_texture=True)
                if struc_r:
                    struc_r.rotation_euler[2] = 3.1415926536
                    transform_apply(struc_r, rotation=True)
                    struc = make_mesh([struc_f, struc_r])
                else:
                    struc = None
            if busstop == 'brt':
                print("xxx")
                brt_f = self.__place_brt_station(seg.right)
                brt_r = self.__place_brt_station(seg.left)
                brt_r.rotation_euler[2] = 3.1415926536
                mirror_uv(brt_r)
                brt_both = make_mesh([duplicate(brt_f), brt_r])
                brt_f.name = str(seg) + '_brt_single'
                brt_both.name = str(seg) + '_brt_both'
                clean_materials(brt_f)
                clean_materials(brt_both)

            lanes_r.rotation_euler[2] = 3.1415926536
            transform_apply(lanes_r, rotation=True)

            lanes = make_mesh([lanes_f, lanes_r])
            lanes.name = str(seg) + '_lanes'
            clean_materials(lanes)
        else:
            lanes, struc = self.__make_segment(seg, mode, busstop)
            lanes.name = str(seg) + '_lanes'
            clean_materials(lanes)
        if struc:
            struc.name = str(seg) + '_structure'
            clean_materials(struc)
        '''
        Note about how the game processes slope models:
        Here right-hand traffic is considered. Left-hand traffic simply
        swaps the upward and downward slopes using the invert flag.
        When creating slope segments, the UPWARD slope (going from tunnel to ground) is
        a FORWARD segment, and the DOWNWARD slope is a BACKWARD segment. To obtain the 
        downward slope, the invert flag of the forward slope is negated. 
        However, the CSUR code builds a DOWNWARD slope by default. This will require
        the slope model of a symmetric road (eg. 4DC, 6DR) to be rotated by 180 degrees
        to give the proper forward slope without requiring flags.
        For asymmetrical roads (eg.2R3-4R3, 4R), the upward and downward slopes are 
        mirror images along the x-axis so they have to be modeled separately.
        The downward slope will always require the invert flag and the upward
        slope will always forbid the invert flag (see segment presets).
        '''
        if mode[0] == 's':
            if isinstance(seg, csur.TwoWay):
                struc.rotation_euler[2] = 3.1415926536
                transform_apply(struc, rotation=True)
            # upward and downward slopes use the same model
            if isinstance(seg, csur.TwoWay) and str(seg.left) == str(seg.right):
                return lanes, struc
            # upward and downward slopes use different models
            else:
                #lanes = make_mirror(lanes, axis=0, copy=False, realign=False)
                if isinstance(seg, csur.TwoWay):
                    struc_up = make_mirror(struc, axis=0, copy=True, realign=False)
                    struc_down = struc
                else:
                    struc_down = make_mirror(struc, axis=0, copy=True, realign=False)
                    struc_up = struc
                return lanes, (struc_up, struc_down)
        if busstop == 'brt':
            return lanes, struc, brt_f, brt_both
        else:
            return lanes, struc

    def make_presentation(self, seg, mode='g'):
        lanes, struc = self.make(seg, mode)
        return make_mesh([lanes, struc])


    def make_node(self, seg, mode, compatibility=False):
        deselect()
        lb = self.lane_border
        margin = 0.15 if mode[0] == 'g' else 0
        # the exact margin should be 0.15, let the node protrude into the curb by 5cm for safety
        if isinstance(seg, csur.TwoWay):
            p = seg.right.start.index(Segment.LANE)
            junction = None
            if not compatibility:
                i_curb = seg.right.start[1:].index(Segment.CURB) + 1
                stopline = place_unit(self.objs['NODE']['stop_line'], 
                            [seg.right.x_start[p], seg.right.x_end[p]], 
                            [seg.right.x_start[i_curb] + lb, seg.right.x_end[i_curb] + lb])
                median = None
                if not seg.undivided and seg.left.x_start == seg.right.x_start \
                    and seg.left.x_end == seg.right.x_end:
                    x0 = seg.right.x_start[seg.right.start.index(Segment.LANE)]
                    median = place_unit(self.objs['NODE']['median'],
                                    [-x0 + lb + margin, -x0 + lb + margin],
                                    [x0 - lb - margin, x0 - lb - margin],
                                    scale_mode=0)         
                junction = make_mesh([stopline, median]) if median else stopline
                reset_origin(junction)
            elements_l = Modeler.make_node(self, seg.left, mode, compatibility)
            elements_r = Modeler.make_node(self, seg.right, mode, compatibility)
            # if the node is asymmetrical then recenter the end of the node
            # use halfcosine interpolation so two consecutive nodes can align
            '''
            cancels node for non-centered roads
            if seg.left.start != seg.right.start:
                w_left = get_dims(pavement_l.data)[0]
                w_right = get_dims(pavement_r.data)[0]
                w_new = max(w_left, w_right)
                pavement_l = place_unit(pavement_l, [0, 0], [w_left, w_new], interpolation='cosine', copy=False)
                pavement_r = place_unit(pavement_r, [0, 0], [w_right, w_new], interpolation='cosine', copy=False)
            '''
            elements = []
            for el, er in zip(elements_l, elements_r):
                if el:
                    el = make_mirror(el, copy=False, realign=False)
                    new_element = make_mesh([el, er])
                    new_element.name = str(seg) + '_' + '_'.join(el.name.split('_')[1:])
                else:
                    new_element = None
                elements.append(new_element)              
            if not compatibility:
                elements[-1] = make_mesh([elements[-1], junction]) if elements[-1] else junction
            return tuple(elements)
        sidewalk = []
        # sidewalk2 does not have crossing
        sidewalk2 = []
        junction = []
        if seg.roadtype() != "b":
            raise NotImplementedError("Node is only valid for base module!")
        else:
            units = [x or y for x, y in zip(seg.start, seg.end)]
            xs_start, xs_end = seg.x_start, seg.x_end
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            # compatibility mode does not have median parts
            if units[p] == Segment.MEDIAN and not compatibility:
                if p != 0:
                    # side median in segment is wider by 0.2m
                    obj = self.objs['NODE']['median']
                    junction.append(place_unit(obj, [xs_start[p] + lb + margin, xs_end[p] + lb + margin],
                                    [xs_start[p + nblocks] - lb - margin, xs_end[p + nblocks] - lb - margin], scale_mode=0))
            elif units[p] in [Segment.CURB, Segment.BARRIER] and (p == 0 or p == len(units) - 1):
                obj = self.objs['NODE']['curb'] if units[p] == Segment.CURB else self.objs['ELEVATED']['barrier']
                if p == 0:
                    obj = make_mirror(obj)
                    sidewalk.append(place_unit(obj, 
                                [xs_start[0], xs_end[0]], 
                                [xs_start[1] - lb - margin, xs_end[1] - lb - margin], copy=False))
                else:
                     obj = duplicate(obj)
                     sidewalk.append(place_unit(obj, 
                            [xs_start[-2] + lb + margin, xs_end[-2] + lb + margin], 
                            [xs_start[-1], xs_end[-1]], copy=False))
            elif units[p] == Segment.SIDEWALK:
                # sidewalk model is determined by whether the road is wider than 3R in g mode
                key = 'sidewalk' if seg.x_start[-1] > 5.5 * LANEWIDTH else 'sidewalk_small'
                if p == 0:
                    obj = make_mirror(self.objs['NODE'][key], axis=0)
                    sidewalk.append(place_unit(obj, 
                            [xs_start[0], xs_end[0]], 
                            [xs_start[2] - lb - margin, xs_end[2] - lb - margin], preserve_obj=True, copy=False))
                    # do not create the no-crossing sidewalk in compatibility mode
                    if not compatibility:
                        obj = make_mirror(self.objs['NODE']['sidewalk2'], axis=0)
                        sidewalk2.append(place_unit(obj, 
                                [xs_start[0], xs_end[0]], 
                                [xs_start[2] - lb - margin, xs_end[2] - lb - margin], preserve_obj=True, copy=False))
                else:
                    sidewalk.append(place_unit(self.objs['NODE'][key], 
                            [xs_start[-3] + lb + margin, xs_end[-3] + lb + margin], 
                            [xs_start[-1], xs_end[-1]], preserve_obj=True))
                    if not compatibility:
                        sidewalk2.append(place_unit(self.objs['NODE']['sidewalk2'], 
                            [xs_start[-3] + lb + margin, xs_end[-3] + lb + margin], 
                            [xs_start[-1], xs_end[-1]], preserve_obj=True))     
            p += nblocks
        # i_curb is the LAST index of CURB in units
        i_curb = len(units) - units[::-1].index(Segment.CURB) - 1
        asphalt_hw = xs_start[i_curb] + lb + margin
        asphalt = place_unit(self.objs['NODE']['asphalt'], [0, 0], [4 * LANEWIDTH, 4 * LANEWIDTH], scale_mode=1)
        asphalt = place_unit(asphalt, 
                    [0, 0], 
                    [asphalt_hw, asphalt_hw], scale_mode=0, preserve_uv=(0 if asphalt_hw >= 4 * LANEWIDTH else -1),
                    copy=False)
        # needs to consider one-way roads with nodes, in this case
        # the road will be always symmetric so the asphalt is just mirrored
        if xs_start[0] + xs_start[-1] == 0 and xs_end[0] + xs_end[-1] == 0:
            asphalt_l = make_mirror(asphalt, copy=True, realign=False)
            #flip_normals(asphalt_l)
            asphalt = make_mesh([asphalt, asphalt_l])
            # normals will be flipped after these operations, need to restore
            #flip_normals(asphalt)
        sidewalk = make_mesh(sidewalk)
        reset_origin(sidewalk)
        if not compatibility:
            if sidewalk2:
                sidewalk2 = make_mesh(sidewalk2)
                reset_origin(sidewalk2)
                sidewalk2.name = str(seg) + "_node_sidewalk2"
            if junction:
                junction = make_mesh(junction)
                reset_origin(junction)
                junction.name = str(seg) + "_node_junction"
            else:
                junction = None
        # make compatibility nodes to connect vanilla roads
        if compatibility:
            place_slope(sidewalk, -0.15 + 3 * EPS, dim=64)
            place_slope(asphalt, -0.3, dim=64)
            sidewalk.name = str(seg) + "_cpnode_sidewalk"
            asphalt.name = str(seg) + "_cpnode_asphalt"
            return sidewalk, asphalt
        else:
            sidewalk.name = str(seg) + "_node_sidewalk"
            asphalt.name = str(seg) + "_node_asphalt"
            return sidewalk, sidewalk2, asphalt, junction

    def __get_dc_components(self, seg, divide_line=False, keep_all=False, unprotect_bikelane=True, 
                            central_channel=False, flip_texture=True):
        units = [x or y for x, y in zip(seg.start, seg.end)]
        # turn off the median in the protected bike lane
        # when unprotect_bikelane is FALSE the median is 
        # generated separately
        if Segment.BIKE in units:
            units[units.index(Segment.BIKE) - 1] = Segment.WEAVE
        objs = self.__make_lanes(units, seg.x_start, seg.x_end, divide_line=divide_line, 
                        central_channel=central_channel, flip_texture=flip_texture)
        objs_extra, struc = self.__make_ground(units, seg.x_start, seg.x_end)
        # remove the sidewalk
        for o in struc:
            delete(o)
        curb = objs_extra.pop()
        delete(curb)
        if not keep_all:
            # also remove the curb beside the sidewalk
            if Segment.BIKE in units:
                bike = objs_extra.pop()
                delete(bike)
        objs.extend(objs_extra)
        # build central median separately
        dc_median = seg.x_start[units.index(Segment.LANE)] + LANEWIDTH / 2
        if units.index(Segment.LANE) == 0:
            dc_median += 0.01
        median = [x for x in objs if x.location[0] < dc_median]
        lanes = [x for x in objs if dc_median <= x.location[0]]
        median = make_mesh(median)
        if lanes:
            lanes = make_mesh(lanes)
            reset_origin(lanes)
        else:
            lanes = None
        reset_origin(median)
        sidemedian = None
        #print(units)
        if not unprotect_bikelane:
            sidemedian = Modeler.make_sidemedian(self, seg, units=units)
            if sidemedian:
                reset_origin(sidemedian)
        return median, lanes, sidemedian
    
    def check_dcnode(self, seg):
        deselect()
        if seg.roadtype() != "b":
            raise ValueError("Node is only valid for base module!")
        if not isinstance(seg, csur.TwoWay):
            raise ValueError("Direct connect node is only valid for two-way segments!")

    # hetrogeneous direct connect rule: narrow -> wide
    def make_dc_node(self, seg, target_median=None, unprotect_bikelane=True):
        self.check_dcnode(seg)
        my_median = [-seg.left.x_start[seg.left.start.index(Segment.LANE)],
                      seg.right.x_start[seg.right.start.index(Segment.LANE)]]
        target_median = target_median or my_median

        # when the road is divided or is the median of DC node does not change
        if target_median is my_median or my_median[0] != my_median[1]:
            median_f, lanes_f, sidemedian_f = self.__get_dc_components(seg.right, divide_line=True, unprotect_bikelane=unprotect_bikelane)
            median_r, lanes_r, sidemedian_r = self.__get_dc_components(seg.left, unprotect_bikelane=unprotect_bikelane)
            for x in [median_r, lanes_r, sidemedian_r]:
                if x:
                    x.rotation_euler[2] = 3.141592654
            median = make_mesh([median_f, median_r])
            lanes = make_mesh([lanes_f, lanes_r])
            sidemedian = make_mesh([sidemedian_f, sidemedian_r]) if sidemedian_f else None
            if my_median[0] != my_median[1]:
                align(median.data)
                median = place_unit(median, [my_median[0] - LANEWIDTH/2, target_median[0] - LANEWIDTH/2],
                                            [my_median[1] + LANEWIDTH/2, target_median[1] + LANEWIDTH/2],
                                            copy=False)
                # prevent z-fighting
                median.location[2] = 3 * EPS
                transform_apply(median, location=True)
            dcnode = make_mesh([median, lanes])
            # remove local lanes in local-express roads
            if len(seg.right.decompose()) == 2:
                low_r = seg.right.decompose()[0].x_right
                high_l = -seg.left.decompose()[0].x_right
                strip(dcnode, low_r, 1 / EPS)
                strip(dcnode, -1 / EPS, high_l)
        # when the road is undivided, must create another segment from factory
        # unprotect bike lanes is set to false because the markings are
        # already not kept straight
        else:
            mode = CSURFactory.infer_ground_variation(seg)
            max_match = 0
            for k, v in CSURFactory.roadside.items():
                if len(v) >= max_match and seg.right.start[-len(v):] == v:
                    mode = k
                    max_match = len(v)
            blocks_f, blocks_r = seg.right.decompose(), seg.left.decompose()
            dcnode_rev = CSURFactory(mode=mode, roadtype='s').get([-my_median[0], -target_median[0]], blocks_f[0].nlanes)
            dcnode_fwd = CSURFactory(mode=mode, roadtype='s').get([target_median[1], my_median[1]], blocks_f[0].nlanes)
            dcnode, sidemedian = Modeler.convert_to_dcnode(self, csur.TwoWay(dcnode_rev, dcnode_fwd), unprotect_bikelane=False)            
        dcnode.name = str(seg) + "_dcnode"
        if sidemedian:
            sidemedian.name = str(seg) + "_dcnode_sidemedian"
        return dcnode, sidemedian

    

    def convert_to_dcnode(self, dcnode_seg, unprotect_bikelane=False):
        central_channel = dcnode_seg.right.start[0] == Segment.CHANNEL or dcnode_seg.right.end[0] == Segment.CHANNEL 
        median_f, lanes_f, sidemedian_f = self.__get_dc_components(dcnode_seg.right, 
                                                        unprotect_bikelane=unprotect_bikelane, keep_all=False, 
                                                        divide_line=True, central_channel=central_channel, flip_texture=True)
        median_r, lanes_r, sidemedian_r = self.__get_dc_components(dcnode_seg.left,
                                                        unprotect_bikelane=unprotect_bikelane, 
                                                        keep_all=False, central_channel=central_channel, flip_texture=False)
        
        # DC node is reversed, so rotate the forward parts
        for x in [median_f, lanes_f, sidemedian_f]:
            if x is not None:
                x.rotation_euler[2] = 3.141592654
        #assert False
        node = make_mesh([median_f, lanes_f, median_r, lanes_r])
        transform_apply(node, rotation=True)
        sidemedian = make_mesh([sidemedian_f, sidemedian_r]) if sidemedian_f else None
        if sidemedian:
            transform_apply(sidemedian, rotation=True)
        #mirror_uv(node)
        return node, sidemedian

    '''
    Restores the symmetry of an asymmetrical road using the side with fewer lanes.
    eg. 5DC>4DR, 2R3-3R>4DR3
    '''
    def make_asym_restore_node(self, seg):
        mode = CSURFactory.infer_ground_variation(seg)
        # asymmetrical segment is place that the right (forward) side has more lanes
        if seg.left.n_lanes()[0] > seg.right.n_lanes()[0]:
            raise ValueError("Asymmetric segment should have more lanes on the right side!")
        if not seg.undivided:
            med = max(seg.left.x_start[seg.left.start.index(Segment.LANE)], seg.right.x_start[seg.right.start.index(Segment.LANE)])
            mediancode = str(int(med // (LANEWIDTH / 2))) * 2
            node, sidemedian = Modeler.make_dc_node(self, seg, target_median=[-med, med], unprotect_bikelane=False)
            node = make_mirror(node, copy=False, realign=False)
            if sidemedian:
                sidemedian = make_mirror(sidemedian, copy=False, realign=False)
        else:
            blocks_f, blocks_r = seg.right.decompose(), seg.left.decompose()
            dcnode_rev = CSURFactory(mode=mode, roadtype='b').get(blocks_r[0].x_left, blocks_r[0].nlanes)
            dcnode_fwd = CSURFactory(mode=mode, roadtype='t').get(
                                    [blocks_r[0].x_left, blocks_f[0].x_left], 
                                    [blocks_r[0].nlanes, blocks_f[0].nlanes],
                                left=(blocks_f[0].x_left!=blocks_r[0].x_left))
            node, sidemedian = Modeler.convert_to_dcnode(self, csur.TwoWay(dcnode_rev, dcnode_fwd))
            mediancode = '11'
        node.name = str(seg) + '_restore_node'
        if sidemedian:
            sidemedian.name = str(seg) + '_restore_node_sidemedian'
        return node, sidemedian, mediancode

    '''
    Makes the inversion node for an asymmetrical segment by moving the median but not changing 
    its width.
    It can be fully inverted (eg. 2R-4R to 4R-2R) or invert by half into a symmetric segment
    (eg. 2R-4R to 3R-3R)
    For undivided asymmetrical segments with odd lane difference, inverting in full is straightforward, while
    inverting by half (moving the yellow line to the center) requires reducing one lane.
    For example, 5DC is inverted by half into a 4DC.
    '''
    def make_asym_invert_node(self, seg, halved=False):
        mode = CSURFactory.infer_ground_variation(seg)
        sidemedian = None
        if seg.left.n_lanes()[0] > seg.right.n_lanes()[0]:
            raise ValueError("Asymmetric segment should have more lanes on the right side!")
        #if seg.undivided and abs(seg.left.n_lanes()[0] - seg.right.n_lanes()[0]) % 2 == 1 and halved:
        #    raise ValueError("Undivided segments with an odd lane difference must be inverted in full!")
        if seg.undivided or seg.left.decompose()[0].x_left > 0 \
            and seg.right.decompose()[0].x_left > 0:
            # for divided with a wide median, we can directly create a transition segment and make it a node
            blocks_f, blocks_r = seg.right.decompose(), seg.left.decompose()
            if halved:
                if seg.undivided:
                    dcnode_rev = CSURFactory(mode=mode, roadtype='s').get([blocks_r[0].x_left, 0], blocks_r[0].nlanes)
                    dcnode_fwd = CSURFactory(mode=mode, roadtype='t').get(
                                        [0, blocks_f[0].x_left], 
                                        [blocks_r[0].nlanes, blocks_f[0].nlanes],
                                    left=(blocks_f[0].x_left!=blocks_r[0].x_left))
                else:
                    dcnode_fwd = CSURFactory(mode=mode, roadtype='b').get(blocks_f[0].x_left, blocks_f[0].nlanes)
                    dcnode_rev = CSURFactory(mode=mode, roadtype='t').get(
                                        [blocks_r[0].x_left, blocks_f[0].x_left], 
                                        [blocks_r[0].nlanes, blocks_f[0].nlanes],
                                    left=(blocks_f[0].x_left!=blocks_r[0].x_left))
                # print(dcnode_rev, dcnode_fwd)
                asym_forward_node, sidemedian = Modeler.convert_to_dcnode(self, csur.TwoWay(dcnode_rev, dcnode_fwd))
                # asym_forward_node = make_mirror(asym_forward_node, copy=False, realign=False)
            else:
                dcnode_fwd = CSURFactory(mode=mode, roadtype='t').get(
                                        [blocks_r[0].x_left, blocks_f[0].x_left], 
                                        [blocks_r[0].nlanes, blocks_f[0].nlanes],
                                    left=True)  
                dcnode_rev = dcnode_fwd
                asym_forward_node, struc = Modeler.make(self, csur.TwoWay(dcnode_rev, dcnode_fwd))
                delete(struc)
                asym_forward_node.rotation_euler[2] = 3.1415926536
            # NOTE: The game will invert the side median mesh for unknown reason
            # if sidemedian:
            #     sidemedian.rotation_euler[2] = 3.1415926536
            transform_apply(asym_forward_node, rotation=True)
            new_median = [blocks_f[0].x_left + LANEWIDTH/2] * 2
            # print(new_median)
        else:
            # for the divided with 1L median case, we first lay down the road surface entirely using lanes
            # for example, put a 7C for 2R3-4R3
            nlanes = int((seg.left.decompose()[0].x_right + seg.right.decompose()[0].x_right) // LANEWIDTH)
            placeholder = CSURFactory(mode='g', roadtype='b').get(-seg.left.decompose()[0].x_right, nlanes)
            if halved and Segment.BIKE in placeholder.units:
                for i in range(1, len(placeholder.units) - 1):
                    if placeholder.units[i - 1] == Segment.BIKE and placeholder.units[i + 1] == Segment.LANE \
                        or placeholder.units[i + 1] == Segment.BIKE and placeholder.units[i - 1] == Segment.LANE:
                        placeholder.units[i] = Segment.WEAVE
            objs = self.__make_lanes(placeholder.units, placeholder.x_start, placeholder.x_end)
            objs_extra, struc = self.__make_ground(placeholder.units, placeholder.x_start, placeholder.x_end)
            # then we add median and adjust its position
            median = put_objects([self.objs['GROUND']['median_h'], self.objs['LANE']['lane_l']])
            median = make_mesh([median, make_mirror(median, realign=False)])
            median.location[2] = 3 * EPS
            align(median.data)
            for x in struc:
                delete(x)
            if halved:
                if Segment.BIKE in placeholder.units:
                    delete(objs_extra.pop())
                    delete(objs_extra.pop(0))
                delete(objs_extra.pop())
                delete(objs_extra.pop(0))
                sidemedian = Modeler.make_sidemedian(self, placeholder)
            objs.extend(objs_extra)
            median_pos = [seg.left.decompose()[0].x_left + LANEWIDTH/2, seg.right.decompose()[0].x_left + LANEWIDTH/2]
            new_median = [(median_pos[0]+median_pos[1])/2]*2 if halved else median_pos 
            place_unit(median, [-median_pos[1], -new_median[0]], [median_pos[0], new_median[1]], copy=False)
            asym_forward_node = make_mesh(objs + [median])
            reset_origin(asym_forward_node)
        asym_forward_node.name = str(seg) + 'invert_node_forward'
        if halved:
            mediancode = ''.join(str(int(max(0, x // (LANEWIDTH / 2) - 1))) for x in new_median)
            asym_forward_node.name = str(seg) + '_expand_node'
            return asym_forward_node, sidemedian, mediancode
        else:
            asym_backward_node = make_mirror(asym_forward_node, realign=False)
            asym_backward_node.name = str(seg) + '_invert_node_backward'
            return asym_forward_node, asym_backward_node

    # DC node for local_express segments, inplemented as a switch 
    # between different local/express combinations, eg. 3+2/4+1
    # can be built using either transition or ramp module
    # tempoarily use ramp module (easier)
    # dlanes: the number of express lanes to increase
    def make_local_express_dc_node(self, seg, dlanes):
        mode = CSURFactory.infer_ground_variation(seg)
        # asymmetrical segment is place that the right (forward) side has more lanes
        if len(seg.right.decompose()) != 2:
            raise ValueError("Not a local-express road!")
        blocks = seg.right.decompose()
        xleft = blocks[0].x_left
        if (dlanes <= -blocks[0].nlanes or dlanes >= blocks[1].nlanes):
            raise ValueError("Invalid local-express dcnode combination!")
        #dlanes=0 keeps the local-express median but unprotects bikelane
        dcnode_fwd = CSURFactory(mode=mode, roadtype='r').get([xleft] * 2, 
                                [[blocks[0].nlanes + dlanes, blocks[1].nlanes - dlanes], [blocks[0].nlanes, blocks[1].nlanes]])
        dcnode_rev = CSURFactory(mode=mode, roadtype='r').get([xleft] * 2, 
                                [[blocks[0].nlanes, blocks[1].nlanes], [blocks[0].nlanes + dlanes, blocks[1].nlanes - dlanes]])                        
        node, sidemedian = Modeler.convert_to_dcnode(self, csur.TwoWay(dcnode_rev, dcnode_fwd), unprotect_bikelane=False)
        pl = pr = 0
        if dlanes == 0:
            pl = dcnode_rev.start[dcnode_rev.start.index(Segment.LANE):].index(Segment.MEDIAN)
            pr = dcnode_fwd.start[dcnode_fwd.start.index(Segment.LANE):].index(Segment.MEDIAN)
            xlow = -dcnode_rev.x_start[pl]
            xhigh = dcnode_fwd.x_start[pr]
        else:
            pl = pr = 0
            while dcnode_fwd.start[pr] == dcnode_fwd.end[pr]:
                pr += 1
            while dcnode_rev.start[pl] == dcnode_rev.end[pl]:
                pl += 1
            xlow = -dcnode_rev.x_start[pl] + LANEWIDTH


            xhigh = dcnode_fwd.x_start[pr] - LANEWIDTH
        strip(node, xlow, xhigh)
        node.name = str(seg) + '_le_dcnode'
        sidemedian.name = str(seg) + '_le_dcnode_sidemedian'
        node.location[2] += 6 * EPS
        transform_apply(node, location=True)
        return node, sidemedian

    '''
    Only makes the side median for a weave segment.
    '''
    def make_sidemedian(self, seg, units=None):
        deselect()
        lb = self.lane_border
        p = 0
        objs = []
        units = units or [x or y for x, y in zip(seg.start, seg.end)]
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.WEAVE:
                if units[p - 1] == Segment.LANE:
                    objs.append(place_unit(self.objs['LANE']['lane_r'],
                                [seg.x_start[p] - LANEWIDTH/2, seg.x_end[p] - LANEWIDTH/2], 
                                [seg.x_start[p] + lb, seg.x_end[p] + lb],
                                ))
                objs.append(place_unit(self.objs['GROUND']['median_f'],
                        [seg.x_start[p] + lb, seg.x_end[p] + lb], 
                        [seg.x_start[p + nblocks] - lb, seg.x_end[p + nblocks] - lb]
                        ))
                if units[p + nblocks + 1] == Segment.LANE:
                    objs.append(place_unit(self.objs['LANE']['lane_l'],
                                [seg.x_start[p + nblocks] - lb, seg.x_end[p + nblocks] - lb],
                                [seg.x_start[p + nblocks] + LANEWIDTH/2, seg.x_end[p + nblocks] + LANEWIDTH/2], 
                                ))        
            p += nblocks
        if len(objs) == 0:
            return None
        obj = make_mesh(objs)
        obj.location[2] += 3 * EPS
        transform_apply(obj, location=True)
        return obj


class ModelerLodded(Modeler):
    def __init__(self, config_file, bridge=False, tunnel=True, optimize=False):
        super().__init__(config_file, bridge, tunnel, lod=False, optimize=optimize)
        self.lodmodeler = Modeler(config_file, bridge, tunnel, lod=True, optimize=optimize)
        self.lod_cache = {}

    def save(self, obj, path):
        try:
            lod_model = self.lod_cache[id(obj)]
        except KeyError:
            raise KeyError("object %s does not have LOD modeled", str(obj), id(obj))
        super().save(obj, path)
        self.lodmodeler.save(lod_model, ''.join(path.split('.')[:-1]) + '_lod.FBX')

    def cachelod(self, model, lod):
        if type(model) != tuple:
            model = (model, )
            lod = (lod, )
        for m, l in zip(model, lod):
            if type(m) == tuple:
                self.cachelod(m, l)
            elif type(l) == bpy.types.Object:
                l.name += '_lod'
                self.lod_cache[id(m)] = l

    def make_arrows(self, seg):
        model = super().make_arrows(seg)
        deselect()
        lod = self.lodmodeler.make_arrows(seg)
        self.cachelod(model, lod)
        return model

    def make_solidlines(self, seg, both=False):
        model = super().make_solidlines(seg, both)
        deselect()
        lod = self.lodmodeler.make_solidlines(seg, both)
        self.cachelod(model, lod)
        return model

    def make_soundbarrier(self, seg):
        model = super().make_soundbarrier(seg)
        deselect()
        lod = self.lodmodeler.make_soundbarrier(seg)
        self.cachelod(model, lod)
        return model

    def make(self, seg, mode='g', busstop=None):
        model = super().make(seg, mode, busstop)
        deselect()
        lod = self.lodmodeler.make(seg, mode, busstop)
        self.cachelod(model, lod)
        return model

    def make_node(self, seg, mode, compatibility=False):
        model = super().make_node(seg, mode, compatibility)
        deselect()
        lod = self.lodmodeler.make_node(seg, mode, compatibility)
        self.cachelod(model, lod)
        return model

    def make_dc_node(self, seg, target_median=None, unprotect_bikelane=True):
        model = super().make_dc_node(seg, target_median, unprotect_bikelane)
        deselect()
        lod = self.lodmodeler.make_dc_node(seg, target_median, unprotect_bikelane)
        self.cachelod(model, lod)
        return model
    
    def make_local_express_dc_node(self, seg, target_median):
        model = super().make_local_express_dc_node(seg, target_median)
        deselect()
        lod = self.lodmodeler.make_local_express_dc_node(seg, target_median)
        self.cachelod(model, lod)
        return model

    def make_asym_restore_node(self, seg):
        model = super().make_asym_restore_node(seg)
        deselect()
        lod = self.lodmodeler.make_asym_restore_node(seg)
        self.cachelod(model, lod)
        return model

    def make_asym_invert_node(self, seg, halved=False):
        model = super().make_asym_invert_node(seg, halved)
        deselect()
        lod = self.lodmodeler.make_asym_invert_node(seg, halved)
        self.cachelod(model, lod)
        return model

    def convert_to_dcnode(self, dcnode_seg, unprotect_bikelane=False):
        model = super().convert_to_dcnode(dcnode_seg, unprotect_bikelane)
        deselect()
        lod = self.lodmodeler.convert_to_dcnode(dcnode_seg, unprotect_bikelane)
        self.cachelod(model, lod)
        return model

    def make_uturn(self, seg):
        model = super().make_uturn(seg)
        deselect()
        lod = self.lodmodeler.make_uturn(seg)
        self.cachelod(model, lod)
        return model

    def make_sidemedian(self, seg):
        model = super().make_sidemedian(seg)
        deselect()
        lod = self.lodmodeler.make_sidemedian(seg)
        self.cachelod(model, lod)
        return model
    
