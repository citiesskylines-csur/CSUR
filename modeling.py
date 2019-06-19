import os
import sys 
import bpy
from mathutils import Vector
from math import cos, pi, factorial


WORKDIR = 'F:/Work/csl/roads/CSUR/'
sys.path.append(WORKDIR) 

import importlib
import model_elements
import csur
importlib.reload(model_elements)
importlib.reload(csur)
from model_elements import *
from csur import Segment, CSURFactory

LANEWIDTH = csur.LANEWIDTH

# wrapper to deselect objects
def selection_safe(f):
    def wrapper(*args, **kwargs):
        [obj.select_set(False) for obj in bpy.data.objects]
        return f(*args, **kwargs)
    return wrapper
            

# Wrap blender ops into function of objects:
def make_func(op, return_selected=False):
    def wrapper(obj, *args, **kwargs):
        hide = obj.hide_get()
        obj.hide_set(False)
        obj.select_set(True)
        ret = op(*args, **kwargs)
        obj.select_set(False)
        obj.hide_set(hide)
        if return_selected:
            ret = bpy.context.selected_objects[0]
            ret.select_set(False)
        return ret
    return wrapper

# wrapped ops
set_origin = make_func(bpy.ops.object.origin_set)
duplicate = make_func(bpy.ops.object.duplicate, return_selected=True)


# equal operator to prevent truncation error
EPS = 1e-6
eq = lambda x, y: abs(x - y) < EPS
deq = lambda x, y: (x - y).length_squared < EPS ** 2


def align(mesh, axis=0, left=True):
    end = min if left else max
    endpoint = end(v.co[axis] for v in mesh.vertices)
    for v in mesh.vertices:
        v.co[axis] -= endpoint

# function aliases        
get_dims = lambda m: [max(v.co[ax] for v in m.vertices) - min(v.co[ax] for v in m.vertices) for ax in range(3)]
    

def link_image(obj, img_bpy):
    obj.material_slots[0].material.node_tree.nodes['Image Texture'].image = img_bpy

#Bezier curve
binom = lambda n, k, x: factorial(n) / (factorial(k) * factorial(n - k)) * x**i * (1-x)**(n-i)
bezier = lambda pts, t: sum(p * binom(len(pts), i, t) for i, p in enumerate(pts))


INTERP_TYPE = 'bezier2'
def interpolate(x0, x1, alpha):
    if INTERP_TYPE == 'linear':
        return x0 + (x1 - x0) * alpha
    if INTERP_TYPE == 'cosine':
        return x0 + (x1 - x0) * (1 - cos(alpha * pi)) / 2
    if 'bezier' in INTERP_TYPE:
        if INTERP_TYPE == 'bezier2':
            u = -2 + 4 * alpha + (5  - 16 * alpha + 16 * alpha**2) ** (1/2)
            b0, b1, b2 = 1/2, -1/2, 1/2
        elif INTERP_TYPE == 'bezier4':
            u = 4 - 8 * alpha + (-11 - 64 * alpha + 64 * alpha**2 ) ** (1/2)
            b0, b1, b2 = 1/2, -3/4 * (1-1.73205j), -1/4 * (1+1.73205j)
        elif INTERP_TYPE == 'bezier8':
            u = 40 - 80 * alpha + (23*5 - 1280*5 * alpha + 1280*5 * alpha**2 ) ** (1/2)
            b0, b1, b2 = 1/2, -7*(1-1.73205j)/(4*5**(1/3)), -(1+1.73205j)/(4*5**(2/3))
        else:
            raise ValueError("Invalid interpolation!")
        t = b0 + b1 * u ** (-1/3) + b2 * u ** (1/3)
        t = t.real
        return x0 + (x1 - x0) * (3 * (1-t) * t**2 + t**3)

@selection_safe   
def load_unit(path, type=CSURUnit.NULL):
    bpy.ops.import_scene.fbx(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.scale = Vector([1, 1, 1])
    obj.location = Vector([0, 0, 0])
    align(obj.data)
    link_image(obj, CSURUnit.textures['d'][type])
    return obj
    
# partitions the vertices of a mesh along an axis:
def partition(mesh, axis=0):
    u_min = min(v.co[axis] for v in mesh.vertices)
    u_max = max(v.co[axis] for v in mesh.vertices)
    u_center = (u_min + u_max) / 2
    return [v for v in mesh.vertices if v.co[axis] <= u_center], \
           [v for v in mesh.vertices if v.co[axis] > u_center]

def place_unit(obj, xs_left, xs_right, copy=True, preserve_uv=False):
    xs_left = xs_left.copy()
    xs_right = xs_right.copy()
    if copy:
        obj = duplicate(obj)
        obj.hide_set(False)
    dims = get_dims(obj.data)
    x0 = xs_left[0]
    obj.location[0] = x0
    if xs_left[0] == xs_left[1] and xs_right[0] == xs_right[1] \
        and eq(xs_right[0] - xs_left[0], dims[0]):
            return obj
    for i in [0, 1]:
        xs_left[i] -= x0
        xs_right[i] -= x0
    vert_l, vert_r = partition(obj.data, axis=0)
    if preserve_uv:
        uv_xdim = min(l.uv[0] for l in obj.data.uv_layers.active.data),\
                    max(l.uv[0] for l in obj.data.uv_layers.active.data)
        uv_ydim = min(l.uv[1] for l in obj.data.uv_layers.active.data),\
                    max(l.uv[1] for l in obj.data.uv_layers.active.data)
        visited = {}
        for face in obj.data.polygons:
            for iv, il in zip(face.vertices, face.loop_indices):
                v = obj.data.vertices[iv]
                l = obj.data.uv_layers.active.data[il]
                alpha = v.co[1] / dims[1] + 0.5
                if v not in visited:
                    dx = interpolate(0, xs_left[1], alpha) if v in vert_l \
                        else interpolate(xs_right[0], xs_right[1], alpha) - dims[0]
                    v.co[0] += dx
                    l.uv[0] -= (dx / dims[0] + int(xs_left[0] > xs_left[1])) * (uv_xdim[1] - uv_xdim[0])
                    visited[v] = l.uv[0]
                else:
                    l.uv[0] = visited[v]              
    else:            
        if xs_left[0] != xs_left[1]:
            for v in vert_l:
                alpha = v.co[1] / dims[1] + 0.5
                dx = interpolate(0, xs_left[1], alpha)
                v.co[0] += dx
        for v in vert_r:
            alpha = v.co[1] / dims[1] + 0.5
            dx = interpolate(xs_right[0], xs_right[1], alpha) - dims[0]
            v.co[0] += dx
    return obj
    
    
@selection_safe
def build_units(start, end, xs_start, xs_end, merge=True, mirror=False):
    units = [x or y for x, y in zip(start, end)]
    lb = CSURUnit.lane_border
    p = 0
    objs_created = []
    while p < len(units):
        nblocks = 1
        while p + nblocks < len(units) and (units[p + nblocks] == units[p] \
                or units[p + nblocks] == Segment.EMPTY):
            nblocks += 1
        if units[p] == Segment.LANE:
            lane_added = 0
            x_left = [xs_start[p] - lb, xs_end[p] - lb]
            x_right = [xs_start[p] + LANEWIDTH / 2, xs_end[p] + LANEWIDTH / 2]
            obj = CSURUnit.objs['lane_l']
            objs_created.append(place_unit(obj, x_left, x_right))
            x_left = x_right.copy()
            lane_added += 1
            for _ in range(nblocks - 1):
                for i, xs in enumerate([xs_start, xs_end]):
                        x_right[i] = min(xs[p + lane_added] + 0.5 * LANEWIDTH,
                                         xs[p + lane_added + 1] - 0.5 * LANEWIDTH)
                    
                obj = CSURUnit.objs['lane_c']
                uvflag = x_left[1] - x_left[0] != x_right[1] - x_right[0]
                objs_created.append(place_unit(obj, x_left, x_right, preserve_uv=uvflag))
                x_left = x_right.copy()
                lane_added += 1
                #print(x_left, x_right)
                
            obj = CSURUnit.objs['lane_r']
            x_right = [x_left[0] + LANEWIDTH / 2 + lb, x_left[1] + LANEWIDTH /2 + lb]
            objs_created.append(place_unit(obj, x_left, x_right))
        elif units[p] == Segment.MEDIAN:
            if xs_start[p] == 0:
                obj = CSURUnit.objs['median_h']
                objs_created.append(place_unit(obj,
                           [xs_start[p], xs_end[p]], 
                           [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))
            else:
                 obj = CSURUnit.objs['median_f']
                 objs_created.append(place_unit(obj,
                           [xs_start[p] + lb, xs_end[p] + lb], 
                           [xs_start[p + nblocks] - lb, xs_end[p + nblocks] - lb]))   
        elif units[p] == Segment.BIKE:
            obj = CSURUnit.objs['bike']
            objs_created.append(place_unit(obj, 
                                [xs_start[p] - lb, xs_end[p] - lb], 
                                [xs_start[p + nblocks] + lb, xs_end[p + nblocks] + lb]))
        elif units[p] == Segment.CURB:
            obj = CSURUnit.objs['curb']            
            objs_created.append(place_unit(obj, 
                                [xs_start[p] + lb, xs_end[p] + lb], 
                                [xs_start[p + nblocks], xs_end[p + nblocks]]))    
        p += nblocks
    if merge:
        obj = make_mesh(objs_created)
        if mirror:
            obj_2 = duplicate(obj)
            obj_2.scale[0]= -1
            obj = make_mesh([obj, obj_2])
    return obj
        
         
@selection_safe
def make_mesh(objs):
    bpy.context.view_layer.objects.active = objs[0]
    [o.select_set(True) for o in objs]
    bpy.ops.object.join()
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.remove_doubles(threshold=EPS)
    bpy.ops.object.editmode_toggle()
    obj = bpy.context.view_layer.objects.active
    #obj.select_set(False)   
    return obj
   

def build(segment, merge=True, mirror=False):
	return build_units(seg.start[seg.first_lane:],
					   seg.end[seg.first_lane:],
					   seg.x_start[seg.first_lane:], 
					   seg.x_end[seg.first_lane:],
					   merge=merge, mirror=mirror)
					
#seg = CSURFactory(mode='g', roadtype='b').get(LANEWIDTH *1.5, 4)
seg = CSURFactory(mode='g', roadtype='t').get([LANEWIDTH*3.5, LANEWIDTH*3.5], [2, 3], left=False)
seg = csur.TwoWay(seg)


CSURUnit.initialize('F:/Work/csl/roads/CSUR/csur_blender.ini')

print(seg.start[seg.first_lane:], seg.end[seg.first_lane:], seg.x_start[seg.first_lane:], seg.x_end[seg.first_lane:])
print(seg)

#seg = CSURFactory(mode='g', roadtype='t').get([LANEWIDTH*1.5, LANEWIDTH*0.5], [3, 4], left=True)
#


build(seg, mirror=True)

#print(CSURUnit.textures['d'][0])
#obj = load_unit('F:/Work/csl/roads/CSUR/models/elem/地面中间带1725.fbx')
#place_unit(CSURUnit.objs['lane_c'], [0, 0], [0, 3.75], preserve_uv=True)


#print(get_dims(obj.data))
