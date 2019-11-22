'''
Manipulates lane props in an road asset.
NOTE: axes use the game convension, X right, Y up, Z front
'''
from copy import deepcopy


'''
Wrapper function to allow operating the function on lists.
'''
def list_op(func):
    def wrapper(obj, *args, **kwargs):
        if isinstance(obj, list):
            for o in obj:
                func(o, *args, **kwargs)
            return obj
        else:
            return func(obj, *args, **kwargs)
    return wrapper


'''
Swaps a pair of substrings in a string.
'''
def swap_substr(s, one, other):
    s = s.replace(one, "$$SUBSTR1$$").replace(other, "$$SUBSTR2$$")
    s = s.replace("$$SUBSTR1$$", other).replace("$$SUBSTR2$$", one)
    return s

'''
Moves the location of the prop along AXIS by a distance of DELTA.
'''
@list_op
def move(prop, delta, axis=0):
    loc_new = float(prop["m_position"]["float"][axis]) + delta
    prop["m_position"]["float"][axis] = str(loc_new)
    return prop

'''
Mirrors the prop so it can be applied to the lane on the other
side of the road. Flips angles, start/end flags, and segment offset
(relative z-position). Used on lane with Both directions.
'''
@list_op
def flip(prop, mirror_position=True):
    prop["m_startFlagsRequired"], prop["m_endFlagsRequired"] \
        = prop["m_endFlagsRequired"], prop["m_startFlagsRequired"]
    prop["m_startFlagsForbidden"], prop["m_endFlagsForbidden"] \
        = prop["m_endFlagsForbidden"], prop["m_startFlagsForbidden"]
    if mirror_position:
        prop["m_position"]["float"][0] = str(-float(prop["m_position"]["float"][0]))
        prop["m_position"]["float"][2] = str(-float(prop["m_position"]["float"][2]))
    prop["m_segmentOffset"] = str(-float(prop["m_segmentOffset"]))
    angle = int(prop["m_angle"])
    angle = angle + 180 if angle <= 0 else angle - 180
    prop["m_angle"] = str(angle)
    # also every occurence of "Start" and "End" in lane flags need to be flipped
    prop["m_flagsRequired"] = swap_substr(prop["m_flagsRequired"], "Start", "End")
    prop["m_flagsForbidden"] = swap_substr(prop["m_flagsForbidden"], "Start", "End")
    return prop

'''
Creates the non-inverted and inverted sets of PROPS.
PROPS should not have any inverted flag. The function 
applies a invert forbidden flag on each prop, and then 
creates a symmetric set of props with invert required flag.
This is used to generate intersection-related props for both left-
and right-hand traffics.
'''
def apply_invert(props):
    # index 0 is non-inverted, index 1 is inverted
    new_props = [[], []]
    fields = ["m_flagsForbidden", "m_flagsRequired"]
    for prop in props:
        if "Inverted" in prop["m_flagsRequired"] + prop["m_flagsForbidden"]:
            raise ValueError("prop list should not contain invert flag to apply inversion!")
        for i, key in enumerate(fields):
            newprop = deepcopy(prop)
            flags = newprop[key].split()
            if len(flags) == 1 and flags[0] == "None":
                flags = ["Inverted"]
            else:
                combined_flag = False
                for j, f in enumerate(flags):
                    if f in ["JoinedJunction", "StartOneWayLeft", "StartOneWayRight", "EndOneWayLeft", "EndOneWayRight"]:
                        flags[j] += "Inverted"
                        combined_flag = True
                if not combined_flag:
                    flags += ["Inverted"]
            newprop[key] = " ".join(flags)
            if i == 1:
                newprop = flip(newprop, mirror_position=False)
                # also need to mirror traffic lights
                if newprop["m_prop"] in ["Traffic Light 01", "Traffic Light 02"]:
                    newprop["m_prop"] += " Mirror"
                elif newprop["m_prop"] in ["Traffic Light 0 Mirror", "Traffic Light 02 Mirror"]:
                    newprop["m_prop"] = newprop["m_prop"].strip(" Mirror")
            new_props[i].append(newprop)
    return tuple(new_props)

'''
Adds a set of PROPS into a LANE at the global position ABS_POS.
If the optional argument HEIGHT is specified, then the absolute height
of the props will be set to HEIGHT.
'''
def add_props(lane, abs_pos, props, height=None):
    if type(props) != list:
        props = [props]
    props = deepcopy(props)
    rel_pos = abs_pos - float(lane["m_position"])
    move(props, rel_pos)
    if height is not None:
        lane_height = float(lane["m_verticalOffset"])
        move(props, height - lane_height, axis=1)
    lane["m_laneProps"]["Prop"].extend(props)
    return lane
        
'''
Adds a set of intersection props into a LANE at the global position ABS_POS.
those props will be added as non-inverted as well as inverted versions
to use on both left- and right-hand traffics.
'''
def add_intersection_props(lane, abs_pos, props):
    props_noinv, props_inv = apply_invert(props)
    return add_props(lane, abs_pos, props_noinv + props_inv)

'''
Flips a lane to the other side of the road.
Returns the flipped lane. Changes the direction of a directed lane
or flips the props of a undirected lane.
'''
def flip_lane(lane, in_place=True):
    if not in_place:
        lane = deepcopy(lane)
    if lane["m_direction"] == "Forward":
        lane["m_direction"] = "Backward"
        lane['m_finalDirection'] = "Backward"
    elif lane["m_direction"] == "Backward":
        lane["m_direction"] = "Forward"
        lane['m_finalDirection'] = "Forward"
    elif lane["m_direction"] == "Both":
        lane["m_laneProps"]["Prop"] = flip(lane["m_laneProps"]["Prop"])
    else:
        raise NotImplementedError
    return lane

'''
Moves a lane along the X-axis to absolute position POS
and keeps the positions of lane props the same.
'''
def move_lane(lane, pos, in_place=True):
    if not in_place:
        lane = deepcopy(lane)
    delta = pos - float(lane['m_position'])
    lane['m_position'] = str(pos)
    for p in lane["m_laneProps"]["Prop"]:
        p["m_position"]["float"][0] = str(float(p["m_position"]["float"][0]) - delta)
    return lane


'''
Combines the lane props from the SOURCE lane
into the TARGET lane.
'''

def combine_props(source, target):
    pass