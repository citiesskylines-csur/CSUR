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
    WALL = 2

    def __init__(self, config_file):
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
        # road_d is the Default texture map
        self.textures = {'d':{}}
        self.textures['d'][Modeler.NULL] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['road_d']))
        self.textures['d'][Modeler.SIDEWALK] = bpy.data.images.load(
                                filepath=os.path.join(self.texpath, self.config['TEX']['sidewalk_d']))
        # wall has the same texture as sidewalk
        self.textures['d'][Modeler.WALL] = self.textures['d'][Modeler.SIDEWALK]

        #load models:
        self.objs = {'LANE': {}, 'GROUND': {}, 'ELEVATED': {}}
        #load lanes
        for k, v in self.config['LANE'].items():
            obj = self.__load(v)
            obj.name = 'CSUR_' + k
            self.objs['LANE'][k] = obj
            obj.hide_set(True)

        for k, v in self.config['ELEVATED'].items():
            obj = self.__load(v)
            if k != 'barrier':  
                obj.name = 'CSUR_elv_' + k
                self.objs['ELEVATED'][k] = obj
                obj.hide_set(True)
            else:
                # create both left and right barriers;
                # FBX imports left barrier
                obj.name = 'CSUR_elv_barrier_L'
                self.objs['ELEVATED']['barrier_l'] = obj
                obj.hide_set(True)
                obj_r = duplicate(obj)
                obj_r.scale[0] = -1
                transform_apply(obj_r, location=True, scale=True, rotation=True)
                align(obj_r.data)
                obj_r.name = 'CSUR_elv_barrier_R'
                self.objs['ELEVATED']['barrier_r'] = obj_r
                obj_r.hide_set(True)

        for k, v in self.config['GROUND'].items():
            if 'bus' not in k:
                if k == 'sidewalk':
                    objtype = Modeler.SIDEWALK
                elif k == 'wall':
                    objtype = Modeler.WALL
                else:
                    objtype = Modeler.NULL
                obj = self.__load(v, type=objtype)
                obj.name = 'CSUR_gnd_' + k
                self.objs['GROUND'][k] = obj
                obj.hide_set(True)
        # load bus stop; need to merge two models
        bus_r = self.__load(self.config['GROUND']['bus_road'], recenter=False)
        bus_s = self.__load(self.config['GROUND']['bus_side'], type=Modeler.SIDEWALK, recenter=False)
        obj = make_mesh([bus_r, bus_s])
        obj.name = 'CSUR_gnd_busstop'
        align(obj.data)
        self.objs['GROUND']['busstop'] = obj
        obj.hide_set(True)
            

   
    def __load(self, name, type=NULL, recenter=True):
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
        return obj

    def __build_lanes(self, units, xs_start, xs_end, busstop=False):
        deselect()
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
                    obj = self.objs['LANE']['lane_f']
                    x_left = [xs_start[p] + lb, xs_end[p] + lb]
                else:
                    obj = self.objs['LANE']['lane_l']
                    x_left = [xs_start[p] - lb, xs_end[p] - lb]
                objs_created.append(place_unit(obj, x_left, x_right))
                x_left = x_right.copy()
                lane_added += 1
                for _ in range(nblocks - 1):
                    for i, xs in enumerate([xs_start, xs_end]):
                            x_right[i] = min(xs[p + lane_added] + 0.5 * LANEWIDTH,
                                            xs[p + lane_added + 1] - 0.5 * LANEWIDTH)
                        
                    obj = self.objs['LANE']['lane_c']
                    uvflag = int(x_left[1] - x_left[0] != x_right[1] - x_right[0])
                    objs_created.append(place_unit(obj, x_left, x_right, preserve_uv=uvflag))
                    x_left = x_right.copy()
                    lane_added += 1
                if units[p + nblocks] == Segment.CHANNEL:
                    obj = self.objs['LANE']['lane_f']
                    x_right = [x_left[0] + LANEWIDTH / 2 - lb, x_left[1] + LANEWIDTH /2 - lb]
                else:
                    if units[p + nblocks:] == CSURFactory.roadside['g'] and busstop:
                        obj = self.objs['LANE']['lane_f']
                    else:
                        obj = self.objs['LANE']['lane_r']
                    x_right = [x_left[0] + LANEWIDTH / 2 + lb, x_left[1] + LANEWIDTH /2 + lb]
                objs_created.append(place_unit(obj, x_left, x_right))
            elif units[p] == Segment.CHANNEL:
                x0 = [xs_start[p] - lb, xs_end[p] - lb]
                x2 = [xs_start[p+nblocks] + lb, xs_end[p+nblocks] + lb]
                x1 = [(x0[0] + x2[0]) / 2, (x0[1] + x2[1]) / 2]
                obj = self.objs['LANE']['channel_l']
                objs_created.append(place_unit(obj, x0, x1))
                obj = self.objs['LANE']['channel_r']
                objs_created.append(place_unit(obj, x1, x2))
            p += nblocks
        return objs_created

    def __build_ground(self, units, xs_start, xs_end, busstop=False):
        deselect()
        lb = self.lane_border
        p = 0
        objs_created = []
        while p < len(units):
            nblocks = 1
            while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                    or units[p + nblocks] == Segment.EMPTY):
                nblocks += 1
            if units[p] == Segment.MEDIAN:
                if xs_start[p] == 0:
                    obj = self.objs['GROUND']['median_h']
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            ))
                else:
                    obj = self.objs['GROUND']['busstop'] if busstop else self.objs['GROUND']['median_f'] 
                    if busstop:
                        nblocks = 4
                    objs_created.append(place_unit(obj,
                            [xs_start[p] + lb, xs_end[p] + lb], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            preserve_obj=busstop))   
            elif units[p] == Segment.BIKE:
                obj = self.objs['GROUND']['bike']
                objs_created.append(place_unit(obj, 
                                    [xs_start[p] - lb, xs_end[p] - lb], 
                                    [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]))
            elif units[p] == Segment.CURB:
                # add a wall to the left end of the road
                if p == 0:
                    obj = self.objs['GROUND']['wall']
                    objs_created.append(place_unit(obj, 
                                    [xs_start[p] + lb, xs_end[p] + lb], 
                                    [xs_start[p] + lb, xs_end[p] + lb]))
                    obj = self.objs['GROUND']['curb']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
                else:
                    obj = self.objs['GROUND']['curb']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            elif units[p] == Segment.SIDEWALK:
                obj = self.objs['GROUND']['sidewalk']
                objs_created.append(place_unit(obj, 
                                    [xs_start[p], xs_end[p]], 
                                    [xs_start[p + nblocks], xs_end[p + nblocks]]))
            p += nblocks
        return objs_created

    def __build_all(self, roadtype, units, x_start, x_end, busstop):
        objs = []
        # place traffic lanes
        objs.extend(self.__build_lanes(units, x_start, x_end, busstop))

        # place ground units
        if 'g' in roadtype:
            objs.extend(self.__build_ground(units, x_start, x_end, busstop))
        # place eleated units
        elif 'e' in roadtype:
            objs.extend(self.__build_elevated(units, x_start, x_end))

        obj = make_mesh(objs)
        reset_origin(obj)
        return obj


    def build(self, seg, roadtype='g', busstop=False):
        #check whether a bus stop can be built
        if seg.start[-4:] != CSURFactory.roadside['g'] and busstop:
            print("Cannot build bus stop on this segment")
            busstop = False
        units = [x or y for x, y in zip(seg.start, seg.end)][seg.first_lane:]
        x_start, x_end = seg.x_start[seg.first_lane:], seg.x_end[seg.first_lane:]
        obj = self.__build_all(roadtype, units, x_start, x_end, busstop)
        reset_origin(obj)
        # If the segment is two-way road then mirror it
        # mirror of one side of the road is the reverse
        if isinstance(seg, csur.TwoWay):
            obj_r = self.__build_all(roadtype, units, x_end, x_start, busstop)
            obj_r.rotation_euler[0] = 0
            obj_r.rotation_euler[1] = 0
            obj_r.rotation_euler[2] = 3.1415926536
            obj = make_mesh([obj, obj_r])
        obj.name = str(seg)
        # reset origin
        reset_origin(obj)
        return obj

    def __build_elevated(self, units, xs_start, xs_end):
        dm, bm, mm = self.deck_margin, self.beam_margin, self.median_margin
        lb = self.lane_border
        bs = get_dims(self.objs['ELEVATED']['beam_sep'].data)[0]
        bw = get_dims(self.objs['ELEVATED']['beam'].data)[0]
        objs_created = []
        # build beams
        w_lanes = max(xs_end[-2] - xs_end[1], xs_start[-2] - xs_start[1])
        w_beam_max = bs + bw
        n_beams = int(w_lanes // (w_beam_max)) + 1
        scale = w_lanes / (w_beam_max * n_beams)
        print(bs, bw, scale)
        beams = []
        if units[0] == Segment.MEDIAN:
            xs_0 = [xs_start[1] - mm, xs_end[1] - mm]
        elif units[0] == Segment.BARRIER:
            xs_0 = [xs_start[0] + bm, xs_end[0] + bm]
        else:
            raise ValueError('Cannot make deck model: not an elevated segment!')
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
        objs_created.append(place_unit(beam_obj, xs_0, [xs_start[-1] - bm, xs_end[-1] - bm], copy=False, scale_all=True))
        # build bridge deck
        obj = self.objs['ELEVATED']['deck_h'] if units[0] == Segment.MEDIAN else self.objs['ELEVATED']['deck_f']
        obj_scaled = duplicate(obj)
        obj_scaled.scale[0] = scale
        transform_apply(obj_scaled, scale=True)
        align(obj_scaled.data, axis=0)
        if units[0] == Segment.MEDIAN:
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
                if xs_start[p] == 0:
                    obj = self.objs['ELEVATED']['median_h']
                    objs_created.append(place_unit(obj,
                            [xs_start[p], xs_end[p]], 
                            [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb],
                            ))
                else:
                    raise ValueError('Cannot built non-central median on elevated road')
            elif units[p] == Segment.BARRIER:
                if p == 0:
                    obj = self.objs['ELEVATED']['barrier_l']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p], xs_end[p]], 
                                        [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
                else:
                    # Loaded model is left barrier
                    obj = self.objs['ELEVATED']['barrier_r']            
                    objs_created.append(place_unit(obj, 
                                        [xs_start[p] + lb, xs_end[p] + lb], 
                                        [xs_start[p + nblocks], xs_end[p + nblocks]]))
            p += nblocks
        return objs_created