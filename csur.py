import re

'''
regex string:
CSUR(-(T|R|S))? ([[1-9]?[0-9]D?(L|S|C|R)[1-9]*P?)+(=|-)?([[1-9]?[0-9]D?(L|S|C|R)[1-9]*P?)*
'''

EPS = 1e-6

LANEWIDTH = 3.75

typename = {'b': 'BASE', 's': 'SHIFT', 't': 'TRANS', 'r': 'RAMP'}

'''
def offset_x(s):
    if s[-1] == 'P':
        return LANEWIDTH * (int(s[:-1]) - 1)
    else:
        return LANEWIDTH * (int(s) - 1.5)


def offset_number(x):
    if abs((x % LANEWIDTH) / LANEWIDTH - 0.5) < EPS:
        return "%d" % round(x / LANEWIDTH + 1.5)
    elif x % LANEWIDTH < EPS or x % LANEWIDTH > LANEWIDTH - EPS:
        return "%dP" % round(x / LANEWIDTH + 1)
    else:
        raise ValueError("Not standardized position of offset %.3f m" % x)
'''

# using new naming scheme
def offset_x(s):
    if s[-1] == 'P':
        return LANEWIDTH * (int(s[:-1]) + 1)
    else:
        return LANEWIDTH * (int(s) + 0.5)


def offset_number(x):
    if abs((x % LANEWIDTH) / LANEWIDTH - 0.5) < EPS:
        return "%d" % round(x / LANEWIDTH - 0.5)
    elif x % LANEWIDTH < EPS or x % LANEWIDTH > LANEWIDTH - EPS:
        return "%dP" % round(x / LANEWIDTH - 1)
    else:
        raise ValueError("Not standardized position of offset %.3f m" % x)


'''def offset_number(x):
    if x not in offsets:
        raise ValueError("Not standardized position of offset %.4f m" % x)
    else:
        return offsets[x]
'''

BLOCK_SEPERATOR = ""
DIRECTION_SEPERATOR = "-"
SEGMENT_END_SEPERATOR = "="

def typecode(roadtype):
    if roadtype == 'b':
        return ""
    else:
        return roadtype.upper()

def splitlist(arr, val):
    arr = arr.copy()
    splited = []
    while val in arr:
        splited.append(arr[:arr.index(val)])
        arr = arr[arr.index(val) + 1:]
    if arr:
        splited.append(arr)
    return splited

def get_name(blocks, reversed=False):
    if reversed:
        blocks = [x[::-1] for x in blocks[::-1]]
    name_start = [str(x) for x in blocks[0]]
    name_end = [str(x) for x in blocks[1]]
    return [name_start, name_end]

def combine_name(names):
    names = names.copy()
    if len(names) != 2:
        raise ValueError('cannot combine name: segment should have 2 ends!')
    names = [DIRECTION_SEPERATOR.join( \
                [BLOCK_SEPERATOR.join(y) for y in splitlist(x, DIRECTION_SEPERATOR)]\
            ) 
            for x in names]
    return names[0] if names[0] == names[1] else SEGMENT_END_SEPERATOR.join(names)

def twoway_reduced_name(block_l, block_r):
    # can reduce same segments in different directions
    block_l_copy = block_l.copy()
    block_r_copy = block_r.copy()
    reduced = []
    i = 0
    while block_l_copy and block_r_copy:
        l, r = block_l_copy.pop(0), block_r_copy.pop(0)
        if l.x_left + r.x_left == 0:
            centered = Carriageway(l.nlanes + r.nlanes, -l.x_right)
            if r.x_left == 0 and r.nlanes - l.nlanes == 1:
                suffix = 'S'
            else:
                suffix = centered.suffix()
            reduced.append("%dD%s" % (centered.nlanes, suffix))
            i += 1
        elif str(l) == str(r):       
            reduced.append("%dD%s" % (2 * l.nlanes, l.suffix()))
            i += 1
    name_l = [str(x) for x in block_l[i:]]
    name_r = [str(x) for x in block_r[i:]]
    if not reduced:
        reduced = [DIRECTION_SEPERATOR]
    return name_l + reduced + name_r

class Segment():
    # length of standard road segment
    LENGTH = 64
    
    # aliases for building units
    EMPTY = 0
    LANE = 1
    MEDIAN = 2
    BIKE = 3
    CURB = 4
    SIDEWALK = 5
    BARRIER = 6
    CHANNEL = 7
    SHOULDER = 8
    WEAVE = 9
    PARKING = 10
    
    # width of each building unit
    widths = [0, 
              LANEWIDTH, 
              LANEWIDTH/2, 
              2.75, 
              0.5, 
              3.75, 
              LANEWIDTH/4, 
              LANEWIDTH/2, 
              LANEWIDTH/2, 
              LANEWIDTH/2, 
              2.75]

    def get_lane_blocks(config):
        p1 = 0
        lanes = []
        while p1 < len(config):
            while p1 < len(config) and config[p1] > Segment.LANE:
                p1 += 1
            if p1 == len(config):
                break
            p2 = p1 + 1
            while config[p2] <= Segment.LANE:
                p2 += 1
            lanes.append([p1, p2])
            p1 = p2
        return lanes
    
    # object methods
    def __init__(self, start, end, x_left=[0, 0]):
        if len(start) != len(end):
            raise ValueError("Cannot create segment: start and end do not align, start is %s but end is %s" % (start, end))
        self.start = start
        self.end = end
        self.x_start = [sum(Segment.widths[c] for c in start[:i]) for i in range(len(start) + 1)]
        self.x_end = [sum(Segment.widths[c] for c in end[:i]) for i in range(len(end) + 1)]
        for i in range(len(start) + 1):
            self.x_start[i] += x_left[0]
            self.x_end[i] += x_left[1]
    
    def width(self):
        return [self.x_start[-1] - self.x_start[0], 
                self.x_end[-1] - self.x_end[0]]
    
    def x_max(self):
        return max(self.x_start[-1], self.x_end[-1])
    
    def n_lanes(self):
        return [sum(c == 1 for c in self.start), sum(c == 1 for c in self.end)]
    
    def decompose(self):
        lanes = [Segment.get_lane_blocks(self.start), 
                 Segment.get_lane_blocks(self.end)]
        decomposed = [[], []]
        cs = [self.start, self.end]
        for i, xs in enumerate([self.x_start, self.x_end]):
            decomposed[i] = [Carriageway(sum(cs[i][l[0]:l[1]]), xs[l[0]]) for l in lanes[i]]
        return decomposed


    def reverse(self):
        return type(self)(self.end, self.start, x_left=[self.x_end[0], self.x_start[0]])
  
    def roadtype(self):
        raise NotImplementedError("Undefined road type!")

    def get_typecode(self):
        return typecode(self.roadtype())

    def middle_index(self):
        return 0

    def __str__(self):
        prefix = "CSUR-%s " % self.get_typecode() if self.get_typecode() else "CSUR "
        return prefix + combine_name(get_name(Segment.decompose(self)))
    
    def __repr__(self):
        return self.__str__()

class StandardWidth:
    LANE = Segment.widths[Segment.LANE]
    MEDIAN = Segment.widths[Segment.MEDIAN]
    BIKE = Segment.widths[Segment.BIKE]
    CURB = Segment.widths[Segment.CURB]
    SIDEWALK = Segment.widths[Segment.SIDEWALK]
    BARRIER = Segment.widths[Segment.BARRIER]
    CHANNEL = Segment.widths[Segment.CHANNEL]
    SHOULDER = Segment.widths[Segment.SHOULDER]
    WEAVE = Segment.widths[Segment.WEAVE]
    PARKING = Segment.widths[Segment.PARKING]

class Carriageway():
    width = Segment.widths[Segment.LANE]
    init_r = Segment.widths[Segment.MEDIAN]
    
    def __init__(self, nlanes, x_left):
        self.nlanes = nlanes
        self.x_left = x_left
        self.x_right = self.x_left + Carriageway.width * self.nlanes
        #if self.x_left == 0 or self.x_right == 0:
        #    raise ValueError("Do not initialize lane border at the origin")
        
    def get_position(self):
        return [self.x_left, self.x_right] 
    
    def get_offset(self):
        return (self.x_left + self.x_right) / 2

    def mirror(self):
        return Carriageway(self.nlanes, -self.x_right)

    def suffix(self):
        if self.get_offset() == 0:
            offset_code = 'C'
            return offset_code
        #elif self.get_offset() == Carriageway.init_r:
        #    offset_code = 'CR'
        #    return offset_code
        #elif self.get_offset() == -Carriageway.init_r:
        #    offset_code = 'CL'
        #    return offset_code
        elif self.get_offset() > 0:
            offset_code = 'R'
        else:
            offset_code = 'L'
        if abs(self.x_left - Carriageway.init_r) < EPS \
                or abs(self.x_right + Carriageway.init_r) < EPS \
                or self.get_offset() == 0:
            n_offset = ''
        else:
            n_offset = offset_number(max(-self.x_left, self.x_right))
        return offset_code + n_offset
    
    def __str__(self):
        return str(self.nlanes) + self.suffix()
          
    def __repr__(self):
        return self.__str__()
    

class BaseRoad(Segment):
    def __init__(self, units, x0):
        #Construct object
        super().__init__(units, units, x_left=[x0, x0])
        self.units = self.start
        self.x = self.x_start
    
    def roadtype(self):
        return 'b'

    #Overrides decompose and __str__ methods
    def decompose(self):
        return [Carriageway(l[1] - l[0], self.x_start[l[0]]) \
                for l in Segment.get_lane_blocks(self.start)]


'''
# creates undivided road segment from the one-way segment
class Undivided(Segment):
    def __init__(self, base_road, first_lane):
        # first lane corresponds to the start configuration
        if first_lanes <= 0:
            raise ValueError("first lane should be positive as to have two directions!")
        elif first_lane[0] > base_road.n_lanes() / 2:
            raise ValueError("there should be more lanes in the x>0 side!")
        blocks = super(Segment, base_road).decompose()
        super().__init__(base_road.start, base_road.end, [base_road.x_start[0], base_road.x_end[0]], 
                         first_lane)
        self.first_lane = first_lane

    def __str__(self):
        split = re.split("(\D+)", self.decompose(full_road=True)[0].__str__())
        if self.first_lane == self.n_lanes()[0] / 2:
            return "CSUR: %sD%s" % (split[0], "".join(split[1:]))
        else:
            return "CSUR: %sDA%s" % (split[0], "".join(split[1:]))

'''
# creates a two-way segment from two one-way segments
class TwoWay(Segment):

    @staticmethod
    def create_median(seg, center, undivided=False): 
        #print(center, seg)
        fill = Segment.CHANNEL if undivided else Segment.MEDIAN
        if isinstance(seg, BaseRoad):
            p = seg.units.index(Segment.LANE)
            units = seg.units.copy()[p:]
            n = int((seg.x[p] - center[0]) // Segment.widths[Segment.MEDIAN])
            #print(seg.x[p], center)
            units = [fill] * n + units
            return BaseRoad(units, center[0])
         
        start = seg.start.copy()
        end = seg.end.copy()
        if start[0] != Segment.MEDIAN or end[0] != Segment.MEDIAN:
            p1 = p2 = 0
            while start[p1] and start[p1] != Segment.LANE:
                p1 += 1
            while end[p2] and end[p2] != Segment.LANE:
                p2 += 1   
            start = start[p1:]
            end = end[p2:]
            n_start = int((seg.x_start[p1] - center[0]) // Segment.widths[Segment.MEDIAN])
            n_end = int((seg.x_end[p2] - center[1]) // Segment.widths[Segment.MEDIAN])
            #print(seg, center, n_start, n_end)
            start = [fill] * n_start + [Segment.EMPTY] * max(n_end - n_start, 0) + start
            end = [fill] * n_end  + [Segment.EMPTY] * max(n_start - n_end, 0) + end
        return type(seg)(start, end, x_left=[center[0], center[1]])

    @staticmethod
    def clean_undivided(seg):
        p1 = p2 = 0
        while seg.start[p1] and seg.start[p1] != Segment.LANE:
            p1 += 1
        while seg.end[p2] and seg.end[p2] != Segment.LANE:
            p2 += 1
        start = seg.start.copy()[p1:]
        end = seg.end.copy()[p2:]
        if type(seg) == BaseRoad:
            return BaseRoad(start, x0=seg.x_start[p1])
        else:
            return type(seg)(start, end, x_left=[seg.x_start[p1], seg.x_end[p2]])

    @staticmethod
    def is_undivided(left, right):
        pl = [x.index(Segment.LANE) for x in [left.start, left.end]]
        pr = [x.index(Segment.LANE) for x in [right.start, right.end]]
        return left.x_start[pl[0]] + right.x_end[pr[1]] == 0 or left.x_end[pl[1]] + right.x_start[pr[0]] == 0

    # initialize using a left and a right segment
    def __init__(self, left, right, append_median=True):
        self.left = left
        self.right = right
        self.undivided = TwoWay.is_undivided(left, right)
        if self.undivided:
            self.left = TwoWay.clean_undivided(self.left)
            self.right = TwoWay.clean_undivided(self.right)
        self.center = [(self.right.x_start[0] - self.left.x_end[0]) / 2, 
                        (self.right.x_end[0] - self.left.x_start[0]) / 2]
        if append_median: 
            if not self.undivided and self.left.x_start[-1] == self.right.x_start[-1] and self.left.x_end[-1] == self.right.x_end[-1]:
                self.center = [min(self.center), min(self.center)]
            self.left = TwoWay.create_median(self.left, [-self.center[1], -self.center[0]], undivided=self.undivided)
            self.right = TwoWay.create_median(self.right, self.center, undivided=self.undivided)
    
    def middle_index(self):
        return len(self.left.start)
  
    def __str__(self):
        typecode = str(self.left.get_typecode()) + str(self.right.get_typecode())
        if len(typecode) == 2 and typecode[0] == typecode[1]:
            typecode = typecode[0]
        prefix = "CSUR-%s " % typecode if typecode else "CSUR "
        blocks_l = Segment.decompose(self.left)
        blocks_r = Segment.decompose(self.right)
        names = [twoway_reduced_name(x, y) for x, y in zip(blocks_l[::-1], blocks_r)]
        return prefix + combine_name(names)

    def roadtype(self):
        typestring = (self.left.roadtype() + self.right.roadtype()).strip("b")
        if len(typestring) > 1 and typestring[0] != typestring[1]:
            raise Exception("Invalid two-way construction!")
        return "b" if typestring == "" else typestring[0]
            


 
class Transition(Segment):
    def roadtype(self):
        return "t" 

class Ramp(Segment):
    def roadtype(self):
        return "r"  

class Shift(Segment):
    def roadtype(self):
        return "s" 

class CSURFactory():
    '''
    Mode:
    g - ground, e - elevated, b - bridge, s - slope, t - tunnel
    ge - ground express lane
    ge - ground compact, w/o bike lanes and sidewalk
    gp - ground parking, w/o bike lanes
    ex - elevated expressway, has a sholder of 0.5L wide
    
    '''
    roadside = {
                # standard ground road
                'g': [Segment.MEDIAN, Segment.BIKE, Segment.CURB, Segment.SIDEWALK],
                # standard road with weaving sections
                'gw': [Segment.MEDIAN, Segment.BIKE, Segment.CURB, Segment.SIDEWALK],
                # ground express lanes
                'ge': [Segment.CURB],
                # compact ground w/o bike lanes:
                'gc': [Segment.CURB, Segment.SIDEWALK],
                # ground road with parking space
                'gp': [Segment.PARKING, Segment.CURB, Segment.SIDEWALK],
                # expressway
                'ex': [Segment.SHOULDER, Segment.BARRIER],
                # elevated 
                'e': [Segment.BARRIER],
                # bridge
                'b': [Segment.SIDEWALK, Segment.BARRIER],
                # slope
                's': [Segment.BARRIER],
                # tunnel
                't': [Segment.BARRIER],  
               }
    road_in = {'g': Segment.CURB, 'ge': Segment.CURB, 'e': Segment.BARRIER,
               'gp': Segment.CURB, 'gc': Segment.CURB, 'gw': Segment.CURB,
                'b': Segment.BARRIER, 'ex': Segment.BARRIER,
                's': Segment.BARRIER, 't': Segment.BARRIER}
    
    def get_units(mode, lane_left, *blocks, n_median=1, prepend_median=False):
        roadside = CSURFactory.roadside[mode]
        units = []
        # left side of road
        if prepend_median and lane_left == Segment.widths[Segment.MEDIAN]:
            units.append(Segment.MEDIAN)
            segment_left = 0
        elif lane_left > -2:
            units.append(CSURFactory.road_in[mode])
            segment_left = lane_left - Segment.widths[CSURFactory.road_in[mode]]
        else:
            units.extend(roadside[::-1])
            segment_left = lane_left - sum(Segment.widths[c] for c in roadside)   
        # traffic lanes
        if isinstance(blocks[0], list):
            blocks = blocks[0]
        for block in blocks:
            units.extend([Segment.LANE] * block)
            units.extend([Segment.MEDIAN] * n_median)
        for _ in range(n_median):
            units.pop()
        #right side of road
        units.extend(roadside)
        return units, segment_left

    def fill_median(left_seg, right_seg, type_override):
        p_left = len(left_seg.start) - 1
        p_right = 0
        while left_seg.start[p_left] != Segment.LANE and left_seg.end[p_left] != Segment.LANE:
            p_left -= 1
        while right_seg.start[p_right] != Segment.LANE and right_seg.end[p_right] != Segment.LANE:
            p_right += 1
        n_start = int((right_seg.x_start[p_right] - left_seg.x_start[p_left + 1]) // StandardWidth.MEDIAN)
        n_end = int((right_seg.x_end[p_right] - left_seg.x_end[p_left + 1]) // StandardWidth.MEDIAN)
        start_new = left_seg.start[:p_left + 1] + [Segment.MEDIAN] * n_start + \
                    [Segment.EMPTY] * max(n_end - n_start, 0) + right_seg.start[p_right:]
        end_new = left_seg.end[:p_left + 1] + [Segment.MEDIAN] * n_end + \
                    [Segment.EMPTY] * max(n_start - n_end, 0) + right_seg.end[p_right:]
        if type_override == 'b':
            segtype = BaseRoad
        elif type_override == 's':
            segtype = Shift
        elif type_override == 't':
            segtype = Transition
        elif type_override == 'r':
            segtype = Ramp
        else:
            raise ValueError("Invalid road type!")
        return segtype(start_new, end_new, x_left=[left_seg.x_start[0], left_seg.x_end[0]])
    
    def __init__(self, mode='e', roadtype='b'):
        self.mode = mode
        self.roadtype = roadtype
        if self.roadtype == 'b':
            self.get = self.get_base
        elif self.roadtype == 't':
            self.get = self.get_transition
        elif self.roadtype == 'r':
            self.get = self.get_ramp
        elif self.roadtype == 's':
            self.get = self.get_shift
        #elif self.roadtype == 'a':
        #    self.get = self.get_access
        
    def get_base(self, lane_left, *blocks, n_median=1):
        units, x0 = CSURFactory.get_units(self.mode, lane_left, *blocks, n_median=n_median)
        return BaseRoad(units, x0)
    
    def get_transition(self, lane_lefts, n_lanes, left=False):
        start, x0_start = CSURFactory.get_units(self.mode, lane_lefts[0], n_lanes[0], prepend_median=False)
        end, x0_end = CSURFactory.get_units(self.mode, lane_lefts[1], n_lanes[1], prepend_median=False)
        if left:
            p, inc = 0, 1
        else:
            p, inc = -1, -1
        while (start[p] != Segment.LANE):
            p += inc
        if not left:
            p += 1
        if n_lanes[0] < n_lanes[1]:
            start = start[:p] + [Segment.EMPTY] * (n_lanes[1] - n_lanes[0]) + start[p:]
        elif n_lanes[0] > n_lanes[1]:
            end = end[:p] + [Segment.EMPTY] * (n_lanes[0] - n_lanes[1]) + end[p:]
        else:
            raise ValueError("No transition between both ends")
        return Transition(start, end, [x0_start, x0_end])

    def get_ramp(self, lane_lefts, n_lanes, n_median=[1, 1]):
        if abs(len(n_lanes[0]) - len(n_lanes[1])) == 2 and lane_lefts[0] == lane_lefts[1] and abs(sum(n_lanes[0]) - sum(n_lanes[1])) == 1:
            # t: which end is the main road
            t = len(n_lanes[0]) > len(n_lanes[1])
            return self.get_access(lane_lefts[0], n_lanes[t][0], n_lanes[1-t][0] + 1, n_lanes[1-t][1], reverse=t)
        start, x0_start = CSURFactory.get_units(self.mode, 
                                                lane_lefts[0], n_lanes[0],
                                                n_median=n_median[0], 
                                                prepend_median=False)
        end, x0_end = CSURFactory.get_units(self.mode, 
                                            lane_lefts[1], n_lanes[1],
                                            n_median=n_median[1],
                                            prepend_median=False)
        p = 0
        while p < len(start):
            while p < len(start) and start[p] == end[p]:
                p += 1
            if p < len(start):
                # do not convert central median to channels
                if start[p] == Segment.MEDIAN and p != 0:
                    start[p] = Segment.CHANNEL
                    end.insert(p, Segment.EMPTY)
                elif end[p] == Segment.MEDIAN and p != 0:
                    end[p] = Segment.CHANNEL
                    start.insert(p, Segment.EMPTY)
                p += 1
            if len(start) + 1 == len(end):
                start.insert(-len(CSURFactory.roadside[self.mode]), Segment.EMPTY)
            if len(end) + 1 == len(start):
                end.insert(-len(CSURFactory.roadside[self.mode]), Segment.EMPTY)
                
        return Ramp(start, end, [x0_start, x0_end])
    
    def get_shift(self, lane_lefts, *blocks, n_median=[1, 1]):
        units, x0 = CSURFactory.get_units(self.mode, lane_lefts[0], *blocks)
        return Shift(units, units, [x0, x0 - lane_lefts[0] + lane_lefts[1]])

    def get_access(self, lane_left, gnd_road, i_a, n_a, reverse=False):
        units, x0 = CSURFactory.get_units(self.mode, lane_left, gnd_road)
        start = units.copy()
        end = units.copy()
        p = 0
        while (start[p] != Segment.LANE):
            p += 1
            
        # locate (i_a)th lane
        p += i_a - 1
        # Replace (n_a + 1) lanes with n_a lanes and 2 channels
        if p + n_a + 1> gnd_road:
            raise ValueError("Not enough lanes to build access road")
        if n_a > 2:
            raise ValueError("too many lanes for access road ramp!")
        if n_a == 2:
            end = end[:p] + [Segment.CHANNEL] + [Segment.LANE, Segment.EMPTY, Segment.LANE] \
                    + [Segment.CHANNEL] + end[p + n_a + 1:]
        else:
            end = end[:p] + [Segment.CHANNEL, Segment.EMPTY] + n_a * [Segment.LANE] \
                    + [Segment.CHANNEL] + end[p + n_a + 1:]
        # Add empty units to instruct segment constructor
        start = start[:p] + [Segment.EMPTY] + start[p:p+n_a+1] + [Segment.EMPTY] + start[p + n_a + 1:]
        if not reverse:
            return Ramp(start, end, x_left=[x0, x0])
        else: 
            return Ramp(end, start, x_left=[x0, x0])