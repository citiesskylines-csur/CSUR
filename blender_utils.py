import bpy
from mathutils import Vector

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
    obj.material_slots[0].material.node_tree.nodes['Image Texture'].image = img_bpy

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
Approximate equal operator to prevent truncation error.
EQ for scaler values and DEQ for vector distances.
'''
EPS = 1e-5
eq = lambda x, y: abs(x - y) < EPS
deq = lambda x, y: (x - y).length_squared < EPS ** 2

'''
Get dimensions of an object along xyz axes.
'''
get_dims = lambda m: [max(v.co[ax] for v in m.vertices) \
            - min(v.co[ax] for v in m.vertices) for ax in range(3)]


'''
Interpolation functions between (x0, x0) and (x1, x1)
LINEAR: linear interpolation
COSINE: cosine function interpolation
BEZIER(N=2,4,8): Bezier curve with control points at 1/N and (1-1/N)
'''

INTERP_TYPE = 'bezier4'

#Parametric bezier curve
binom = lambda n, k, x: factorial(n) / (factorial(k) * factorial(n - k)) * x**i * (1-x)**(n-i)
bezier = lambda pts, t: sum(p * binom(len(pts), i, t) for i, p in enumerate(pts))

def interpolate(x0, x1, alpha, interp_type=INTERP_TYPE):
    if interp_type == 'linear':
        return x0 + (x1 - x0) * alpha
    if interp_type == 'cosine':
        return x0 + (x1 - x0) * (1 - cos(alpha * pi)) / 2
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
Make a list of objects OBJ into a single mesh.
Also merges overlapping vertices.
'''
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
IF PRESERVE_UV then the UV mapping does not transform when moving vertices 
(same behavior as vertex silde)
'''
def place_unit(obj, xs_left, xs_right, copy=True, preserve_uv=0, preserve_obj=False, scale_all=False):
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
                    if scale_all:
                        # assume that object is x-aligned
                        beta = v.co[0] / dims[0]
                        x1 = interpolate(0, xs_right[0], beta, 'linear')
                        x2 = interpolate(xs_left[1], xs_right[1], beta, 'linear')
                        dx = interpolate(x1, x2, alpha) - dims[0] * beta
                    else:
                        if v in vert_l:
                            dx = interpolate(0, xs_left[1], alpha)
                        elif v in vert_c:
                            dx = interpolate(xs_right[0] / 2, (xs_left[1] + xs_right[1]) / 2, alpha) - dims[0] / 2
                        else:
                            dx = interpolate(xs_right[0], xs_right[1], alpha) - dims[0]
                    v.co[0] += dx
                    if preserve_uv in [-1, 1]:
                        l.uv[0] -= preserve_uv \
                                        * (dx / dims[0] + int(xs_left[0] > xs_left[1])) \
                                        * (uv_xdim[1] - uv_xdim[0])
                        visited[v] = l.uv[0]
                    elif preserve_uv in [-2, 2]:
                        l.uv[1] -= (preserve_uv // 2) \
                                        * (dx / dims[0] + int(xs_left[0] > xs_left[1])) \
                                        * (uv_ydim[1] - uv_ydim[0])
                        visited[v] = l.uv[1]
                else:
                    if preserve_uv in [-1, 1]:
                        l.uv[0] = visited[v]
                    elif preserve_uv in [-2, 2]:
                        l.uv[1] = visited[v]            
    else:
        if scale_all:
            # assume that object is x-aligned
            for v in obj.data.vertices:
                alpha = v.co[1] / dims[1] + 0.5
                beta = v.co[0] / dims[0]
                x1 = interpolate(0, xs_right[0], beta, 'linear')
                x2 = interpolate(xs_left[1], xs_right[1], beta, 'linear')
                dx = interpolate(x1, x2, alpha) - dims[0] * beta
                v.co[0] += dx
        else:  
            if xs_left[0] != xs_left[1]:
                for v in vert_l:
                    alpha = v.co[1] / dims[1] + 0.5
                    dx = interpolate(0, xs_left[1], alpha)
                    v.co[0] += dx
            for v in vert_c:
                alpha = v.co[1] / dims[1] + 0.5
                dx = interpolate(xs_right[0] / 2, (xs_left[1] + xs_right[1]) / 2, alpha) - dims[0] / 2
                v.co[0] += dx
            for v in vert_r:
                alpha = v.co[1] / dims[1] + 0.5
                dx = interpolate(xs_right[0], xs_right[1], alpha) - dims[0]
                v.co[0] += dx
    return obj