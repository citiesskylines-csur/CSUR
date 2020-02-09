import bpy
from mathutils import Vector
from math import pi, cos, sin

'''
Decoreator to deselect all objects before function call.
Wrapped on functions using Blender active object ops
'''
def selection_safe(f):
    def wrapper(*args, **kwargs):
        [obj.select_set(False) for obj in bpy.data.objects]
        return f(*args, **kwargs)
    return wrapper

'''
Function to Deselect all objects
Use this instead of decorator for class methods.
'''
deselect = lambda: [obj.select_set(False) for obj in bpy.data.objects]

'''
Aligns a mesh MESH to its endpoint along AXIS.
'''
def align(mesh, axis=0, left=True):
    end = min if left else max
    endpoint = end(v.co[axis] for v in mesh.vertices)
    for v in mesh.vertices:
        v.co[axis] -= endpoint
'''
Links the image texture node of OBJ to Blender image object IMG_BPY.
'''
def link_image(obj, img_bpy):
    try:
        obj.material_slots[0].material.node_tree.nodes['Image Texture'].image = img_bpy
    except IndexError:
        print("This object does not have texture: %s" % obj)

'''
Wrap blender op on active object into a function 
taking the object as the argument.
Can either return the original op output or the selected objects.
'''
def op_to_func(op, return_selected=False):
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

set_origin = op_to_func(bpy.ops.object.origin_set)
duplicate = op_to_func(bpy.ops.object.duplicate, return_selected=True)
transform_apply = op_to_func(bpy.ops.object.transform_apply)

'''
Deletes an object.
'''
def delete(obj):
    hide = obj.hide_get()
    obj.hide_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()
    

'''
Approximate equal operator to prevent truncation error.
EQ for scalar values and DEQ for vector distances.
'''
EPS = 1e-3
eq = lambda x, y: abs(x - y) < EPS
deq = lambda x, y: (x - y).length_squared < EPS ** 2

'''
Get dimensions of an object along xyz axes.
'''
get_dims = lambda m: [max(v.co[ax] for v in m.vertices) \
            - min(v.co[ax] for v in m.vertices) for ax in range(3)]


'''
A piecewise continuous function from (0, 0) to (1, 1)
constructed by two cosine functions at the ends and a 
stright line in between.
'''
phi = lambda x: (1 - cos(x * pi)) / 2
dphi = lambda x: pi / 2 * sin(x * pi)

def cosine_straight(alpha, t):
    k = 1 / ((1 - 2 * t) * dphi(t) + 2 * phi(t))
    if alpha < t:
        return k * phi(alpha)
    elif alpha < 1 - t:
        return k * dphi(t) * (alpha - t) + k * phi(t)
    else:
        return 1 - k * phi(1 - alpha)



'''
Interpolation functions between (x0, x0) and (x1, x1)
LINEAR: linear interpolation
COSINE: cosine function interpolation
BEZIER(N=2,4,8): Bezier curve with control points at 1/N and (1-1/N)
'''

INTERP_TYPE = 'bezier4'

#Parametric bezier curve
binom = lambda n, k, x: factorial(n) / (factorial(k) * factorial(n - k)) * x**k * (1-x)**(n-k)
bezier = lambda pts, t: sum(p * binom(len(pts) - 1, i, t) for i, p in enumerate(pts))

def interpolate(x0, x1, alpha, interp_type=INTERP_TYPE):
    if interp_type == 'linear':
        return x0 + (x1 - x0) * alpha
    if interp_type == 'cosine':
        return x0 + (x1 - x0) * (1 - cos(alpha * pi)) / 2
    if 'cosinestraight' in interp_type:
        t = float(interp_type[len('cosinestraight'):])
        return x0 + (x1 - x0) * cosine_straight(alpha, t)
    if interp_type == 'halfcosine':
        return x0 + (x1 - x0) * (1 - cos(alpha * pi / 2))
    if 'bezier' in interp_type:
        if interp_type == 'bezier2':
            u = -2 + 4 * alpha + (5  - 16 * alpha + 16 * alpha**2) ** (1/2)
            b0, b1, b2 = 1/2, -1/2, 1/2
        elif interp_type == 'bezier4':
            u = 4 - 8 * alpha + (-11 - 64 * alpha + 64 * alpha**2 ) ** (1/2)
            b0, b1, b2 = 1/2, -3/4 * (1-1.73205j), -1/4 * (1+1.73205j)
        elif interp_type == 'bezier8':
            u = 40 - 80 * alpha + (23*5 - 1280*5 * alpha + 1280*5 * alpha**2 ) ** (1/2)
            b0, b1, b2 = 1/2, -7*(1-1.73205j)/(4*5**(1/3)), -(1+1.73205j)/(4*5**(2/3))
        else:
            raise ValueError("Invalid interpolation!")
        t = b0 + b1 * u ** (-1/3) + b2 * u ** (1/3)
        t = t.real
        return x0 + (x1 - x0) * (3 * (1-t) * t**2 + t**3)

'''
Partitions the vertices of a mesh along an axis.
Mesh should be symmetric along the axis
'''
def partition(mesh, axis=0, return_center=False):
    u_min = min(v.co[axis] for v in mesh.vertices)
    u_max = max(v.co[axis] for v in mesh.vertices)
    u_center = (u_min + u_max) / 2
    if not return_center:
        return [v for v in mesh.vertices if v.co[axis] <= u_center], \
            [v for v in mesh.vertices if v.co[axis] > u_center]
    else:
        return [v for v in mesh.vertices if v.co[axis] <= u_center - EPS], \
            [v for v in mesh.vertices if eq(v.co[axis], u_center)], \
            [v for v in mesh.vertices if v.co[axis] >= u_center + EPS]


'''
makes the mirror image an object along the normal plane of an axis.
returns a duplicate if COPY=TRUE
Also need to flip normals after applying transform
'''
def make_mirror(obj, axis=0, copy=True, realign=True):
    if copy:
        obj = duplicate(obj)
    obj.scale[axis] = -1
    transform_apply(obj, scale=True)
    deselect()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.editmode_toggle()
    if realign:
        align(obj.data, axis=0)
    return obj


def strip(obj, low, high, axis=0):
    deselect()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")   
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in obj.data.vertices: 
        if low <= v.co[axis] <= high:
            v.select = True
        else:
            v.select = False
    bpy.ops.object.mode_set(mode="EDIT") 
    bpy.ops.mesh.delete(type='VERT')  
    bpy.ops.object.editmode_toggle()
    return obj

def invert(obj, axis=2, copy=True, realign=True):
    if copy:
        obj = duplicate(obj)
    obj.rotation_euler[axis] = 3.1415926536
    transform_apply(obj)
    if realign:
        align(obj.data)
    return obj

'''
Make a list of objects OBJS into a single mesh.
Also merges overlapping vertices.
'''
@selection_safe
def make_mesh(objs, merge=True):
    # return None if there is nothing in objs
    if not objs:
        return None
    bpy.context.view_layer.objects.active = objs[0]
    [o.select_set(True) for o in objs if o]
    bpy.ops.object.join()
    bpy.ops.object.editmode_toggle()
    if merge:
        bpy.ops.mesh.remove_doubles(threshold=EPS)
    bpy.ops.object.editmode_toggle()
    obj = bpy.context.view_layer.objects.active
    obj.select_set(False)   
    
    return obj

'''
Cleans the UV mapping of the object into [0, 1].
Periodic condition of the UV coordinates is assumed.
To ensure all UV coorainates are within [0, 1], a face 
should not be unwrapped across the border of the texture images.
eg., from u=-0.2 to u=0.2.
If originally the uv coordinates exceeds [0, 1], eg., from u=-0.2 to u=1.8,
then do not change the UV mapping.
'''
def clean_uv(obj):
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
        # check if the uv already exceed [0, 1]
        uv_dim = [max(obj.data.uv_layers.active.data[il].uv[i] \
                    for il in face.loop_indices) \
                 - min(obj.data.uv_layers.active.data[il].uv[i] \
                    for il in face.loop_indices) \
                 for i in [0, 1]]
        for il in face.loop_indices:
            l = obj.data.uv_layers.active.data[il]
            if uv_dim[0] <= 1:
                l.uv[0] += du
            if uv_dim[1] <= 1:
                l.uv[1] += dv

'''
Mirrors the UV coordinates of the object along an axis as 
(u, v) -> (1-u, v) or (u, v) -> (u, 1-v)
'''
def mirror_uv(obj, axis=1):
    for face in obj.data.polygons:
        for il in face.loop_indices:
            l = obj.data.uv_layers.active.data[il]
            l.uv[axis] = 1 - l.uv[axis]

'''
Sets the origin of the object to the scene origin.
'''
def reset_origin(obj):
    for v in obj.data.vertices:
        v.co += obj.location
    obj.location = Vector([0, 0, 0])

'''
Places the objects at the polygon defined by XS_LEFT and XS_RIGHT.
if COPY then the object is duplicated and the duplicate is transformed and placed.

If PRESERVE_UV then the UV mapping does not transform when moving vertices 
(same behavior as vertex silde)

If PRESERVE_OBJ then the object will not be deformed and its left side will be placed a
t xs_left[0]. Used on placing bus stations because they are not specified standard widths.

SCALE_MODE controls how the object is deformed.
0: the vertices to the left of the center are snapped along xs_left, those to the right are
   snapped along xs_right, the vertices at the center are snapped along (xs_left + xs_right) / 2.
   Used on most road pieces like central median and car lanes.
1: the object is scaled as a whole. Used on structural pieces like slope arches and elevated decks.
2: same behavior as 1 but only places non-rectangular units. Used on placing triangular channelizing lines.
'''
def place_unit(obj, xs_left, xs_right, copy=True, preserve_uv=0, preserve_obj=False, scale_mode=0, interpolation=INTERP_TYPE):
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
    if preserve_obj:
        return obj
    for i in [0, 1]:
        xs_left[i] -= x0
        xs_right[i] -= x0
    vert_l, vert_c, vert_r = partition(obj.data, axis=0, return_center=True)
    if preserve_uv:
        deltas = {}
        for v in obj.data.vertices:
            alpha = v.co[1] / dims[1] + 0.5
            if scale_mode:
                # assume that object is x-aligned
                if scale_mode == 1:
                    beta = v.co[0] / dims[0]
                else:
                    raise ValueError("non-parallel scaling conflicts with UV perservation!")
                x1 = interpolate(0, xs_right[0], beta, 'linear')
                x2 = interpolate(xs_left[1], xs_right[1], beta, 'linear')
                dx = interpolate(x1, x2, alpha, interpolation) - dims[0] * beta
            else:
                if v in vert_l:
                    dx = interpolate(0, xs_left[1], alpha, interpolation)
                elif v in vert_c:
                    dx = interpolate(xs_right[0] / 2, (xs_left[1] + xs_right[1]) / 2, alpha, interpolation) - dims[0] / 2
                else:
                    dx = interpolate(xs_right[0], xs_right[1], alpha, interpolation) - dims[0]
            v.co[0] += dx
            deltas[v] = dx
        for face in obj.data.polygons:
            uv_xdim = min(obj.data.uv_layers.active.data[il].uv[0] for il in face.loop_indices),\
                    max(obj.data.uv_layers.active.data[il].uv[0] for il in face.loop_indices)
            uv_ydim = min(obj.data.uv_layers.active.data[il].uv[1] for il in face.loop_indices),\
                        max(obj.data.uv_layers.active.data[il].uv[1] for il in face.loop_indices)
            for iv, il in zip(face.vertices, face.loop_indices):
                v = obj.data.vertices[iv]
                l = obj.data.uv_layers.active.data[il]
                dx = deltas[v]
                if preserve_uv in [-1, 1]:
                    l.uv[0] -= preserve_uv \
                                    * (dx / dims[0] + int(xs_left[0] > xs_left[1])) \
                                    * (uv_xdim[1] - uv_xdim[0])
                elif preserve_uv in [-2, 2]:
                    l.uv[1] -= (preserve_uv // 2) \
                                    * (dx / dims[0] + int(xs_left[0] > xs_left[1])) \
                                    * (uv_ydim[1] - uv_ydim[0])
    else:
        if scale_mode:
            x_min = {}
            x_max = {}
            # assume that object is x-aligned
            if scale_mode == 2:
                for v in obj.data.vertices:
                    if round(v.co[1], 3) not in x_min:
                        vert_same_y = [u.co[0] for u in obj.data.vertices if eq(v.co[1], u.co[1])]
                        x_min[round(v.co[1], 3)] = min(vert_same_y)
                        x_max[round(v.co[1], 3)] = max(vert_same_y)
            for v in obj.data.vertices:
                alpha = v.co[1] / dims[1] + 0.5
                if scale_mode == 1:
                    beta = v.co[0] / dims[0]
                    x1 = interpolate(0, xs_right[0], beta, 'linear')
                    x2 = interpolate(xs_left[1], xs_right[1], beta, 'linear')
                    v.co[0] = interpolate(x1, x2, alpha, interpolation)
                elif scale_mode == 2:
                    xmin_cur = x_min[round(v.co[1], 3)]
                    xmax_cur = x_max[round(v.co[1], 3)]
                    if eq(xmin_cur, xmax_cur):
                        beta = eq(v.co[0], xmax_cur)
                    else:
                        beta = (v.co[0] - xmin_cur) / (xmax_cur - xmin_cur)     
                        x1 = interpolate(0, xs_right[0], beta, 'linear')
                        x2 = interpolate(xs_left[1], xs_right[1], beta, 'linear')
                        #print(x1, x2, beta, alpha)
                        v.co[0] = interpolate(x1, x2, alpha, interpolation)
                else:
                    raise ValueError("invalid scale mode!")
        else:  
            if xs_left[0] != xs_left[1]:
                for v in vert_l:
                    alpha = v.co[1] / dims[1] + 0.5
                    dx = interpolate(0, xs_left[1], alpha, interpolation)
                    v.co[0] += dx
            for v in vert_c:
                alpha = v.co[1] / dims[1] + 0.5
                dx = interpolate(xs_right[0] / 2, (xs_left[1] + xs_right[1]) / 2, alpha, interpolation) - dims[0] / 2
                v.co[0] += dx
            for v in vert_r:
                alpha = v.co[1] / dims[1] + 0.5
                dx = interpolate(xs_right[0], xs_right[1], alpha, interpolation) - dims[0]
                v.co[0] += dx
    return obj

'''
place the object as a slope along the Y axis 
starting from z=0 to z=z_end.
'''
@selection_safe
def place_slope(obj, z_end, interpolation='cosine', dim=None):
    dim = dim or get_dims(obj.data)[1]
    for v in obj.data.vertices:
        alpha = v.co[1] / dim + 0.5
        dz = interpolate(0, z_end, alpha, interpolation)
        v.co[2] += dz
    return obj

'''
put multiple objects together in a row and join them
'''
def put_objects(objs):
    x0 = [0, 0]
    objs_new = []
    for o in objs:
        objs_new.append(place_unit(o, x0, x0, preserve_obj=True))
        dim = get_dims(o.data)[0]
        x0[0] += dim
        x0[1] += dim
    return make_mesh(objs_new)
  

'''
Cleans duplicate materials
https://blender.stackexchange.com/questions/75790/how-to-merge-around-300-duplicate-materials
'''
@selection_safe
def clean_materials(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # only search on own object materials
    unique_materials = {}
    remove_slots = []
    try:
        mat_list = [x.material.name for x in bpy.context.object.material_slots]
    except AttributeError:
        return

    # the following only works in object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    for s in bpy.context.object.material_slots:
        name = s.material.name
        if name[:-4] not in unique_materials:
            unique_materials[name[:-4]] = name
        if name[-3:].isnumeric():
            # the last 3 characters are numbers
            if name[:-4] in unique_materials and unique_materials[name[:-4]] != name:
                index_clean = mat_list.index(unique_materials[name[:-4]])
                index_wrong = mat_list.index(name)

                # get the faces which are assigned to the 'wrong' material
                faces = [x for x in bpy.context.object.data.polygons if x.material_index == index_wrong]

                for f in faces:
                    f.material_index = index_clean

                remove_slots.append(s.name)
    # now remove all empty material slots:
    for s in remove_slots:
        if s in [x.name for x in bpy.context.object.material_slots]:
            bpy.context.object.active_material_index = [x.material.name for x in bpy.context.object.material_slots].index(s)
            bpy.ops.object.material_slot_remove()

@selection_safe
def clean_normals(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

@selection_safe
def flip_normals(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')


def cleanup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # TODO: clean materials

def wipe_materials():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)