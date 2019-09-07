import configparser, os, importlib
import bpy
from mathutils import Vector
import blender_utils
importlib.reload(blender_utils)
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
    NODE = 2
    ELEVATED = 3
    BRIDGE = 4
    TUNNEL = 5
    SLOPE = 6
    

    def __init__(self, config_file, bridge=False, tunnel=True, lod=False):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        # lane_border indicates the width that a lane unit extends at the edge
        self.lane_border = float(self.config['PARAM']['lane_border'])
        self.deck_margin = float(self.config['PARAM']['deck_margin'])
        self.beam_margin = float(self.config['PARAM']['beam_margin'])
        self.median_margin = float(self.config['PARAM']['median_margin'])
        # load textures
        self.texpath = os.path.join(self.config['PATH']['workdir'], 
                                    self.config['PATH']['tex'])
        self.bridge = bridge
        self.tunnel = tunnel
        self.lod = lod
        # road_d is the Default texture map
        self.textures = {'d':{}}
        self.textures['d'][Modeler.NULL] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['road_d']))
        self.textures['d'][Modeler.NODE] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['node_d']))
        self.textures['d']['BRT'] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['brt_d']))

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
        self.objs = {'LANE': {}, 'GROUND': {}, 'NODE': {}, 'ELEVATED': {}, 'BRIDGE': {}, 'TUNNEL': {}, 'SLOPE': {}}
        #load lanes
        for k, v in self.config['LANE'].items():
            obj = self.__load(v)
            obj.name = 'CSUR_' + k
            self.objs['LANE'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['ELEVATED'].items():
            obj = self.__load(v, type=Modeler.ELEVATED)
            obj.name = 'CSUR_elv_' + k
            self.objs['ELEVATED'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['GROUND'].items():
            if k in ['sidewalk', 'wall', 'bus_side']:
                objtype = Modeler.SIDEWALK
                obj = self.__load(v, type=objtype, recenter=(k != 'bus_side'))
            elif k == 'brt_station':
                obj = self.__load(v, type='BRT', recenter=False)
            else:
                objtype = Modeler.NULL
                obj = self.__load(v)
            obj.name = 'CSUR_gnd_' + k
            self.objs['GROUND'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['NODE'].items():
            obj = self.__load(v, type=Modeler.NODE)
            obj.name = 'CSUR_node_' + k
            self.objs['NODE'][k] = obj
            obj.hide_set(True)
        
        if self.bridge:
            for k, v in self.config['BRIDGE'].items():
                obj = self.__load(v, type=Modeler.BRIDGE)
                obj.name = 'CSUR_bdg_' + k
                self.objs['BRIDGE'][k] = obj
                obj.hide_set(True)
        if self.tunnel:
            for k, v in self.config['TUNNEL'].items():
                obj = self.__load(v, type=Modeler.TUNNEL)
                obj.name = 'CSUR_tun_' + k
                self.objs['TUNNEL'][k] = obj
                obj.hide_set(True)
            for k, v in self.config['SLOPE'].items():
                obj = self.__load(v, type=Modeler.TUNNEL, recenter=False)
                obj.name = 'CSUR_slp_' + k
                self.objs['SLOPE'][k] = obj
                obj.hide_set(True)

        if self.lod:
            for k in self.objs.keys():
                for obj in self.objs[k].values():
                    obj.name += '_lod'

    def check_mode(self, mode):
        if mode[0] == 'b' and not self.bridge:
            raise Exception("Bridge mode not loaded!")
        if mode[0] in ['t', 's'] and not self.tunnel:
            raise Exception("Tunnel mode not loaded!")
                
   
    def __load(self, name, type=NULL, recenter=True):
        if self.lod:
            lod_name = ''.join(name.split('.')[:-1] + ['_lod.'] + [name.split('.')[-1]])
            if os.path.exists(os.path.join(self.config['PATH']['workdir'],
                            self.config['PATH']['units'], lod_name)):
                name = lod_name
        path = os.path.join(self.config['PATH']['workdir'],
                            self.config['PATH']['units'],
                            name)
        bpy.ops.import_scene.fbx(filepath=path)
        obj = bpy.context.selected_objects[0]
        obj.animation_data_clear()
        obj.scale = Vector([1, 1, 1])
        obj.location = Vector([0, 0, 0])
        obj.rotation_euler = Vector([0, 0, 0])
        if recenter:
            align(obj.data)
        link_image(obj, self.textures['d'][type])
        # clean UV, using periodic boundary condition
        for face in obj.data.polygons:
            du = dv = 0
            for il in face.loop_indices:
                l = obj.data.uv_layers.active.data[il]
                if du == 0 and l.uv[0] > 1:
                    du = -int(l.uv[0])
                elif du == 0 and l.uv[0] < 0:
                    du = int(l.uv[0]) + 1
                if dv == 0 and l.uv[1] > 1:
                    dv = -int(l.uv[1])
                elif dv == 0 and l.uv[1] < 0:
                    dv = int(l.uv[1]) + 1
            for il in face.loop_indices:
                l = obj.data.uv_layers.active.data[il]
                l.uv[0] += du
                l.uv[1] += dv  
        return obj

    def save(self, obj, path):
        deselect()
        obj.select_set(True)
        bpy.ops.export_scene.fbx(filepath=path, 
                axis_forward='Z', axis_up='Y', use_selection=True, bake_space_transform=True)

    def __make_lanes(self, units, xs_start, xs_end, busstop=None, divide_line=False, central_channel=False):
        deselect()
        xs_start = xs_start.copy()
        xs_end = xs_end.copy()
        units = units.copy()
        lb = self.lane_border
        p = 0
        objs_created = []
        if divide_line and units[0] == Segment.LANE:
            obj = self.objs['LANE']['divide_line']
            objs_created.append(place_unit(obj, [xs_start[0]-lb, xs_end[0]-lb], [xs_start[0]+lb, xs_end[0]+lb]))
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.LANE:
                centered_trans_offset = 0
                if (xs_start[p] - xs_end[p]) * (xs_start[p + nblocks] - xs_end[p + nblocks]) < 0:
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
                    obj = self.objs['LANE']['lane_r']
                    obj_temp = place_unit(obj, [0, 0], [get_dims(obj.data)[0] - lb*1.25, get_dims(obj.data)[0] - lb*1.25], preserve_uv=1)
                    make_mirror(obj_temp, copy=False)
                    x_left = [xs_start[p], xs_end[p]]
                    objs_created.append(place_unit(obj_temp, x_left, x_right, copy=False))
                elif units[p - 1] == Segment.WEAVE:
                    obj = self.objs['LANE']['lane_f']
                    x_left = [xs_start[p] - LANEWIDTH / 4 + lb, xs_end[p] - LANEWIDTH / 4 + lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
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
      
                    uvflag = int(x_left[1] - x_left[0] != x_right[1] - x_right[0])
                    if j == (nblocks - 2) and centered_trans_offset:
                        x_temp = [max(x_right), max(x_right)]
                        obj = self.objs['LANE']['lane_c']
                        objs_created.append(place_unit(obj, x_left, x_temp, preserve_uv=uvflag))
                        x_left = x_temp.copy()
                    else:
                        if (x_left[0] == x_left[1] or x_right[0] == x_right[1]) or (x_left[1] - x_left[0] == x_right[1] - x_right[0]):
                            obj = self.objs['LANE']['lane_c']
                        else:
                            obj = self.objs['LANE']['lane_f']
                            uvflag = 0
                        objs_created.append(place_unit(obj, x_left, x_right, preserve_uv=uvflag))    
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
                    obj = self.objs['LANE']['lane_r']
                    obj_temp = place_unit(obj, [0, 0], [get_dims(obj.data)[0] - lb*1.25, get_dims(obj.data)[0] - lb*1.25],
                                preserve_uv=1)
                    x_right = [x_left[0] + LANEWIDTH / 2, x_left[1] + LANEWIDTH /2]
                    objs_created.append(place_unit(obj_temp, x_left, x_right, copy=False))
                elif units[p + nblocks] == Segment.WEAVE:
                    obj = self.objs['LANE']['lane_f']
                    x_right = [x_left[0] + 3 * LANEWIDTH / 4 - lb, x_left[1] + 3 * LANEWIDTH /4 - lb]
                    objs_created.append(place_unit(obj, x_left, x_right))
                else:
                    if units[p + nblocks:] == CSURFactory.roadside['g'] and busstop in ['single', 'double']:
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
                    if divide_line:
                        obj = self.objs['LANE']['channel']
                        w = get_dims(obj.data)[0]
                        if x0[0] == x2[0]:
                            obj_temp = place_unit(obj, [0, -w], [0, w], 
                                                preserve_uv=1, interpolation='linear')
                        elif x0[1] == x2[1]:
                            obj_temp = place_unit(obj, [0, w], [2*w, w+0.001], 
                                                preserve_uv=1, interpolation='linear')
                        else:
                            obj_temp = place_unit(obj, [0, 0], [2*w, 2*w], 
                                                preserve_uv=1, interpolation='linear')
                        objs_created.append(place_unit(obj_temp, [2*x0[0]-x2[0],2*x0[1]-x2[1]], x2, scale_mode=2, copy=False))
                else:
                    obj = self.objs['LANE']['channel']
                    obj_temp = None 
                    print(x0, x1, x2)
                    if x0[0] == x2[0]:
                        obj_temp = place_unit(obj, [0,0], [0.001,get_dims(obj.data)[0]], preserve_uv=1, interpolation='linear')
                    elif x0[1] == x2[1]:
                        obj_temp = place_unit(obj, [0,0], [get_dims(obj.data)[0], 0.001], preserve_uv=1, interpolation='linear')
                    objs_created.append(place_unit(obj_temp, x1, x2, scale_mode=2))
                    obj_temp = make_mirror(obj_temp, copy=False)
                    objs_created.append(place_unit(obj_temp, x0, x1, scale_mode=2, copy=False))
            elif units[p] == Segment.SHOULDER:
                obj = self.objs['LANE']['lane_f']
                x_left = [xs_start[p] + lb, xs_end[p] + lb]
                if units[p + 1] == Segment.LANE:
                    x_right = [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]
                else:
                    x_right = [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]
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
                    if busstop in ['single', 'double'] and Segment.MEDIAN not in units[p+nblocks:]:
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
                else:
                    obj = self.objs['GROUND']['curb']            
                    lanes_extra.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            elif units[p] == Segment.SIDEWALK:
                obj = self.objs['GROUND']['sidewalk']
                struc.append(place_unit(obj, 
                                    [xs_start[p], xs_end[p]], 
                                    [xs_start[p + nblocks], xs_end[p + nblocks]]))
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

    def __make_tunnel(self, units, xs_start, xs_end):
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
            
            p += nblocks
        
        p = 0
        while units[p] in [Segment.MEDIAN, Segment.BARRIER]:
            p += 1
        obj = self.objs['TUNNEL']['roof']
        objs_created.append(place_unit(obj, [xs_start[p] - lb, xs_end[p] - lb], [xs_start[-2] + lb, xs_end[-2] + lb]))
        return objs_created

    def __make_slope(self, units, xs_start, xs_end, reverse=False):
        objs_created = []
        lb = self.lane_border
        p = 0
        seam = float(self.config['PARAM']['slope_median_seam'])
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
        w_lanes = max(xs_end[-2] - xs_end[1], xs_start[-2] - xs_start[1])
        obj = duplicate(self.objs['SLOPE']['arch'] if w_lanes > 3 * LANEWIDTH else self.objs['SLOPE']['arch2'])
        if reverse:
            obj = make_mirror(obj, axis=1, copy=False)
        objs_created.append(place_unit(obj, 
                            [xs_start[p] - lb, xs_end[p] - lb], [xs_start[-2] + lb, xs_end[-2] + lb], scale_mode=1, copy=False))
        return objs_created

    def __make_elevated(self, units, xs_start, xs_end):
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
                    x0 = [min([xs_start[p] + lb, xs_end[p] + lb])] * 2
                    x1 = [max([xs_start[p+nblocks] - lb, xs_end[p+nblocks] - lb])] * 2
                    obj = duplicate(self.objs['ELEVATED']['gore'])
                    if xs_start[p] == xs_start[p + nblocks]:
                        objs_created.append(place_unit(obj, x0, x1, copy=False))
                    elif xs_end[p] == xs_end[p + nblocks]:
                        obj = make_mirror(obj, axis=1, realign=False)
                        objs_created.append(place_unit(obj, x0, x1, copy=False))              
            p += nblocks
        return objs_created

    def __check_busstop(self, seg, busstop):
        if seg.start[-4:] != CSURFactory.roadside['g'] and busstop:
            raise ValueError("Cannot make bus stop on this segment")

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
                brt.location[2] = 0.15
                reset_origin(brt)
                return brt
            p += nblocks


    def __make_segment(self, seg, mode, busstop, divide_line=False):
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
        print(units)
        # place traffic lanes
        lanes = self.__make_lanes(units, x_start, x_end, busstop, divide_line)
        # place ground units
        if mode[0] == 'g':
            lanes_extra, struc = self.__make_ground(units, x_start, x_end, busstop)
            lanes.extend(lanes_extra)
        # place elevated units
        elif mode[0] == 'e':
            struc = self.__make_elevated(units, x_start, x_end)
        elif mode[0] == 'b':
            struc = self.__make_bridge(units, x_start, x_end, divide_line)
        elif mode[0] == 't':
            struc = self.__make_tunnel(units, x_start, x_end)
        elif mode[0] == 's':
            struc = self.__make_slope(units, x_start, x_end, reverse=(not divide_line))
        lanes = make_mesh(lanes)
        reset_origin(lanes)
        # do not merge bridge mesh to preserve double-sided faces
        if struc:
            struc = make_mesh(struc, merge=mode[0] != 'b')
            reset_origin(struc)
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
        lanes_r = self.__make_lanes(uleft, xsleft, xeleft, None if busstop == 'single' else busstop, central_channel=central_channel)

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
            struc = self.__make_slope(units, x_start, x_end)
        
        lanes_f = make_mesh(lanes_f)
        lanes_r = make_mesh(lanes_r)
        reset_origin(lanes_f)
        reset_origin(lanes_r)

        if struc:
            struc = make_mesh(struc, merge=mode[0] != 'b')
            reset_origin(struc)
        return lanes_f, lanes_r, struc
        

 
    '''
    Note on bus stop configuration:
    None: no bus stop is placed
    'single': a regular bus stop is placed on the right side of the road
    'double': a regular bus stop is placed on both sides of the road
    'brt': a BRT stop is placed on the non-central median of the road;
       the median should be two units (one lane width) wide.
    '''
    def make(self, seg, mode='g', busstop=None):
        busstop = busstop and busstop.lower()
        self.check_mode(mode)
        if isinstance(seg, csur.TwoWay):
            if seg.undivided:
                lanes_f, lanes_r, struc = self.__make_undivided(seg, mode, busstop)     
            else:
                lanes_f, struc_f = self.__make_segment(seg.right, mode, busstop, divide_line=True)
                lanes_r, struc_r = self.__make_segment(seg.left, mode, None if busstop == 'single' else busstop)
                if struc_r:
                    struc_r.rotation_euler[2] = 3.1415926536
                    transform_apply(struc_r, rotation=True)
                    struc = make_mesh([struc_f, struc_r])
                else:
                    struc = None
            if busstop == 'brt':
                #brt_f = self.__place_brt_station(seg.right, reverse=False)
                #brt_r = self.__place_brt_station(seg.left, reverse=True)
                #brt_r.location[0] *= -1
                #brt_r.rotation_euler[2] = 3.1415926536
                #brt = make_mesh([brt_f, brt_r])
                brt = self.__place_brt_station(seg.right, reverse=False)
                brt.name = str(seg) + '_brt'
                #clean_materials(brt)

            lanes_r.rotation_euler[2] = 3.1415926536
            transform_apply(lanes_r, rotation=True)
            lanes_f.name = str(seg) + '_lanes_forward'
            lanes_r.name = str(seg) + '_lanes_backward'
            clean_materials(lanes_f)
            clean_materials(lanes_r)
            lanes = [lanes_f, lanes_r]
            if seg.roadtype() != 'r':
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
        # Slope meshes are reversed
        if mode[0] == 's':
            lanes.rotation_euler[2] = 3.1415926536
            transform_apply(lanes, rotation=True)
            struc.rotation_euler[2] = 3.1415926536
            transform_apply(struc, rotation=True)
        if busstop == 'brt':
            return lanes, struc, brt
        else:
            return lanes, struc

    def make_presentation(self, seg, mode='g'):
        lanes, struc = self.make(seg, mode)
        return make_mesh([lanes, struc])


    def make_node(self, seg):
        deselect()
        margin = 0.1
        if isinstance(seg, csur.TwoWay):
            p = 0
            while seg.right.start[p] == Segment.MEDIAN:
                p += 1
            stopline = place_unit(self.objs['NODE']['stop_line'], 
                        [seg.right.start[p], seg.right.end[p]], 
                        [seg.right.x_start[-3], seg.right.x_end[-3]])
            pavement_l, junction_l = self.make_node(seg.left)
            pavement_r, junction_r = self.make_node(seg.right)
            # if the node is asymmetric then recenter the end of the node
            # use halfcosine interpolation so two consecutive nodes can align
            if seg.left.start != seg.right.start:
                w_left = get_dims(pavement_l.data)[0]
                w_right = get_dims(pavement_r.data)[0]
                w_new = (w_left + w_right) / 2
                pavement_l = place_unit(pavement_l, [0, 0], [w_left, w_new], interpolation='halfcosine', copy=False)
                pavement_r = place_unit(pavement_r, [0, 0], [w_right, w_new], interpolation='halfcosine', copy=False)
            pavement_l = make_mirror(pavement_l, copy=False, realign=False)
            junction_l = make_mirror(junction_l, copy=False, realign=False)
            pavement = make_mesh([pavement_l, pavement_r])
            junction = make_mesh([junction_l, junction_r, stopline])
            pavement.name = str(seg) + "_node_pavement"
            junction.name = str(seg) + "_node_junction"
            return pavement, junction
        lb = self.lane_border
        pavement = []
        junction = []
        if seg.roadtype() != "b":
            raise ValueError("Node is only valid for base module!")
        else:
            units = [x or y for x, y in zip(seg.start, seg.end)]
            xs_start, xs_end = seg.x_start, seg.x_end
        p = 0
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN:
                if p == 0:
                    obj = self.objs['NODE']['central_median']
                    junction.append(place_unit(obj, [xs_start[p], xs_end[p]],
                                [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb], scale_mode=1))
                else:
                    # side median in segment is wider by 0.2m
                    obj = self.objs['NODE']['side_median']
                    junction.append(place_unit(obj, [xs_start[p] + lb + margin, xs_end[p] + lb + margin],
                                    [xs_start[p + nblocks] - lb - margin, xs_end[p + nblocks] - lb - margin], scale_mode=0))
            elif units[p] == Segment.SIDEWALK:
                obj = self.objs['NODE']['sidewalk']
                if p == 0:
                    obj = make_mirror(obj, axis=0)
                    pavement.append(place_unit(obj, 
                            [xs_start[0], xs_end[0]], 
                            [xs_start[2] - lb - margin, xs_end[2] - lb - margin], preserve_obj=True, copy=False))
                else:
                    pavement.append(place_unit(obj, 
                            [xs_start[-3] + lb + margin, xs_end[-3] + lb + margin], 
                            [xs_start[-1], xs_end[-1]], preserve_obj=True))
                
            p += nblocks
        pavement.append(place_unit(self.objs['NODE']['asphalt'], 
                    [0, 0], 
                    [xs_start[-3] + lb + margin, xs_end[-3] + lb + margin], scale_mode=1))
        pavement = make_mesh(pavement)
        reset_origin(pavement)
        junction = make_mesh(junction)
        reset_origin(junction)
        pavement.name = str(seg) + "_node_pavement"
        junction.name = str(seg) + "_node_junction"
        return pavement, junction

    def __get_dc_components(self, seg, divide_line=False, unprotect_bikelane=True, central_channel=False):
        units = [x or y for x, y in zip(seg.start, seg.end)]
        # turn off the median in the protected bike lane
        if unprotect_bikelane and Segment.BIKE in units:
            units[units.index(Segment.BIKE) - 1] = Segment.WEAVE
        objs = self.__make_lanes(units, seg.x_start, seg.x_end, divide_line=divide_line, central_channel=central_channel)
        objs_extra, struc = self.__make_ground(units, seg.x_start, seg.x_end)
        # remove the sidewalk
        for o in struc:
            delete(o)
        # also remove the curb beside the sidewalk
        curb = objs_extra.pop()
        delete(curb)
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
        lanes = make_mesh(lanes)
        reset_origin(median)
        reset_origin(lanes)
        return median, lanes
    
    def check_dcnode(self, seg):
        deselect()
        if seg.roadtype() != "b":
            raise ValueError("Node is only valid for base module!")
        if not isinstance(seg, csur.TwoWay):
            raise ValueError("Direct connect node is only valid for two-way segments!")

    # hetrogeneous direct connect rule: narrow -> wide
    def make_dc_node(self, seg, target_median=None, unprotect_bikelane=True):
        self.check_dcnode(seg)
        print('dcnode:', seg)
        median_f, lanes_f, = self.__get_dc_components(seg.right, divide_line=True, unprotect_bikelane=unprotect_bikelane)
        median_r, lanes_r, = self.__get_dc_components(seg.left, unprotect_bikelane=unprotect_bikelane)
        for x in [median_r, lanes_r]:
            x.rotation_euler[2] = 3.141592654
        median = make_mesh([median_f, median_r])
        lanes = make_mesh([lanes_f, lanes_r])
        my_median = [-seg.left.x_start[seg.left.start.index(Segment.LANE)],
                      seg.right.x_start[seg.right.start.index(Segment.LANE)]]
        target_median = target_median or my_median
        if my_median != [0, 0]:
            align(median.data)
            median = place_unit(median, [my_median[0] - LANEWIDTH/2, target_median[0] - LANEWIDTH/2],
                                        [my_median[1] + LANEWIDTH/2, target_median[1] + LANEWIDTH/2],
                                        copy=False)
        # prevent z-fighting
        median.location[2] = 0.005
        transform_apply(median, location=True)
       
        dcnode = make_mesh([median, lanes])
        dcnode.name = str(seg) + "_dcnode"
        return dcnode

    # DC node which restores the symmetry using the side with fewer lanes
    # eg. 5DC>4DR, 2R3-3R>4DR3
    def make_asym_restore_node(self, seg):
        mode = 'g'
        # asymmetric segment is place that the right (forward) side has more lanes
        if seg.left.n_lanes()[0] > seg.right.n_lanes()[0]:
            raise ValueError("Asymmetric segment should have more lanes on the right side!")

        blocks_f = seg.right.decompose()
        blocks_r = seg.left.decompose()
        dcnode_rev = CSURFactory(mode=mode, roadtype='b').get(blocks_r[0].x_left, blocks_r[0].nlanes)
        dcnode_fwd = CSURFactory(mode=mode, roadtype='t').get(
                                [blocks_r[0].x_left, blocks_f[0].x_left], 
                                [blocks_r[0].nlanes, blocks_f[0].nlanes],
                                left=(blocks_f[0].x_left!=blocks_r[0].x_left))
        if not seg.undivided:
            #print(dcnode_seg.right, dcnode_seg.right.start, dcnode_seg.right.end, dcnode_seg.right.x_start, dcnode_seg.right.x_end)
            #assert False
            med = max(seg.left.x_start[seg.left.start.index(Segment.LANE)], seg.right.x_start[seg.right.start.index(Segment.LANE)])
            mediancode = str(int(med // (LANEWIDTH / 2))) * 2
            node = self.make_dc_node(seg, target_median=[-med, med], unprotect_bikelane=False)
            node = make_mirror(node, copy=False, realign=False)
        else:
            dcnode_seg = csur.TwoWay(dcnode_rev, dcnode_fwd)
            #print(dcnode_seg.right, dcnode_seg.right.start, dcnode_seg.right.end, dcnode_seg.right.x_start, dcnode_seg.right.x_end)
            #assert False
            central_channel = dcnode_seg.right.start[0] == Segment.CHANNEL or dcnode_seg.right.end[0] == Segment.CHANNEL 
            median_f, lanes_f, = self.__get_dc_components(dcnode_seg.right, 
                                                            unprotect_bikelane=False, divide_line=True, central_channel=central_channel)
            median_r, lanes_r, = self.__get_dc_components(dcnode_seg.left,
                                                            unprotect_bikelane=False, central_channel=central_channel)
            # DC node is reversed, so rotate the forward parts
            for x in [median_f, lanes_f]:
                x.rotation_euler[2] = 3.141592654
            #assert False
            node = make_mesh([median_f, lanes_f, median_r, lanes_r])
            transform_apply(node, rotation=True)
            node.name = str(seg) + 'restore_node'
            mediancode = '11'
        return node, mediancode
        
class ModelerLodded(Modeler):
    def __init__(self, config_file, bridge=False, tunnel=True):
        super().__init__(config_file, bridge, tunnel, lod=False)
        self.lodmodeler = Modeler(config_file, bridge, tunnel, lod=True)
        self.lod_cache = {}

    def save(self, obj, path):
        lod_model = self.lod_cache[id(obj)]
        super().save(obj, path)
        self.lodmodeler.save(lod_model, ''.join(path.split('.')[:-1]) + '_lod.FBX')

    def cachelod(self, model, lod):
        if type(model) != tuple:
            model = (model, )
            lod = (lod, )
        for m, l in zip(model, lod):
            self.lod_cache[id(m)] = l

    def make(self, seg, mode='g', busstop=None):
        model = super().make(seg, mode, busstop)
        lod = self.lodmodeler.make(seg, mode, busstop)
        self.cachelod(model, lod)
        return model

    def make_node(self, seg):
        model = super().make_node(seg)
        lod = self.lodmodeler.make_node(seg)
        self.cachelod(model, lod)
        return model

    def make_dc_node(self, seg, target_median=None, unprotect_bikelane=True):
        model = super().make_dc_node(seg, target_median, unprotect_bikelane)
        lod = self.lodmodeler.make_dc_node(seg, target_median, unprotect_bikelane)
        self.cachelod(model, lod)
        return model

    def make_asym_restore_node(self, seg):
        model = super().make_asym_restore_node(seg)
        lod = self.lodmodeler.make_asym_restore_node(seg)
        self.cachelod(model, lod)
        return model


    
