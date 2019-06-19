#from matplotlib import pyplot as plt

DPI = 150

LANEWIDTH = 3.75

offsets = {LANEWIDTH * 3.5: '5', 
           LANEWIDTH * 4: '5P', 
           LANEWIDTH * 5: '6P', 
           LANEWIDTH * 6: '7P', 
           LANEWIDTH * 7: '8P'}

def offset_x(s):
    if s[-1] == 'P':
        return LANEWIDTH * (int(s[:-1]) - 1)
    else:
        return LANEWIDTH * (int(s) - 1.5)

def offset_number(x):
    if abs((x % LANEWIDTH) / LANEWIDTH - 0.5) < 1e-4:
        return "%d" % (x / LANEWIDTH + 1.5)
    elif x % LANEWIDTH == 0:
        return "%dP" % (x / LANEWIDTH + 1)
    else:
        raise ValueError("Not standardized position of offset %.3f m" % x)

'''def offset_number(x):
    if x not in offsets:
        raise ValueError("Not standardized position of offset %.4f m" % x)
    else:
        return offsets[x]
'''
    
    
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
    
    # width of each building unit
    widths = [0, LANEWIDTH, LANEWIDTH/2, 3.05, 0.5, 2.75, LANEWIDTH/4]
    # colors of each building unit
    colors = ["1", 
              "0.3", 
              [0.5, 0.55, 0.48], 
              "0.5", 
              [0.3, 0.77, 0.25], 
              "0.7",
              "0.7"]
    '''
    # static helper functions
    def plot_polygon(ax, xs, dx, **kwargs):
        points = [[xs[0], 0], [xs[1], Segment.LENGTH], [xs[1] + dx[1], Segment.LENGTH], [xs[0] + dx[0], 0]]
        return ax.add_patch(plt.Polygon(points, **kwargs))
    
    def dashed_line(ax, xs, line_part=[0, 1], **kwargs):
        return ax.plot([xs[0] + (xs[1] - xs[0]) * line_part[0], xs[0] + (xs[1] - xs[0]) * line_part[1]],
                       [Segment.LENGTH * line_part[0], Segment.LENGTH * line_part[1]],
                       color="1", ls='--', dashes=(10, 8))
    '''
    def get_lane_blocks(config, first_lane):
        p1 = first_lane
        lanes = []
        while p1 < len(config):
            while p1 < len(config) and config[p1] > 1:
                p1 += 1
            if p1 == len(config):
                break
            p2 = p1 + 1
            while config[p2] <= 1:
                p2 += 1
            lanes.append([p1, p2])
            p1 = p2
        return lanes
    
    # object methods
    def __init__(self, start, end, x_left=[0, 0], first_lane=0):
        if len(start) != len(end):
            raise ValueError("Cannot create segment: start and end do not align")
        self.start = start
        self.end = end
        self.first_lane = first_lane
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
        lanes = [Segment.get_lane_blocks(self.start, self.first_lane), 
                 Segment.get_lane_blocks(self.end, self.first_lane)]
        decomposed = [[], []]
        cs = [self.start, self.end]
        for i, xs, in enumerate([self.x_start, self.x_end]):
            decomposed[i] = [Carriageway(sum(cs[i][l[0]:l[1]]), xs[l[0]]) for l in lanes[i]]
        return decomposed
    
    
    def __str__(self):
        names = self.decompose()
        name_start = "".join([str(x) for x in names[0]])
        name_end = "".join([str(x) for x in names[1]])
        return name_start + '=' + name_end
    
    def __repr__(self):
        return self.__str__()
    '''    
    def draw(self, ax=None):
        if not ax:
            plt.figure(dpi=DPI)
            ax = plt.gca()
        ax.set_aspect('equal', 'box')
        # Draw polygons
        for x0, c0, x1, c1 in zip(self.x_start[:-1], self.start, self.x_end[:-1], self.end):
            Segment.plot_polygon(ax, [x0, x1], [Segment.widths[c0], Segment.widths[c1]],
                                 color=Segment.colors[c0 or c1])
        for i in range(1, len(self.start)):
            if (self.start[i - 1] or self.end[i - 1]) == 1 and (self.start[i] or self.end[i]) == 1:
                line_part = [0, 1]
                if not (self.start[i - 1] and self.start[i]):
                    line_part[0] = 0.5
                if not (self.end[i - 1] and self.end[i]):
                    line_part[1] = 0.5    
                Segment.dashed_line(ax, [self.x_start[i], self.x_end[i]], line_part)
        if not ax:
            plt.show()
    '''
class StandardWidth:
    LANE = Segment.widths[Segment.LANE]
    MEDIAN = Segment.widths[Segment.MEDIAN]
    BIKE = Segment.widths[Segment.BIKE]
    CURB = Segment.widths[Segment.CURB]
    SIDEWALK = Segment.widths[Segment.SIDEWALK]
    BARRIER = Segment.widths[Segment.BARRIER]


class Carriageway():
    width = Segment.widths[Segment.LANE]
    init_r = Segment.widths[Segment.MEDIAN]
    
    def __init__(self, nlanes, x_left):
        self.nlanes = nlanes
        self.x_left = x_left
        self.x_right = self.x_left + Carriageway.width * self.nlanes
        if self.x_left == 0 or self.x_right == 0:
            raise ValueError("Do not initialize lane border at the origin")
        
    def get_position(self):
        return [self.x_left, self.x_right]
    
    def get_offset(self):
        return (self.x_left + self.x_right) / 2
    
    def __str__(self):
        if self.get_offset() == 0:
            offset_code = 'C'
        elif self.get_offset() > 0:
            offset_code = 'R'
        else:
            offset_code = 'L'
        if self.x_left == Carriageway.init_r \
                or self.x_right == -Carriageway.init_r \
                or self.get_offset() == 0:
            n_offset = ''
        else:
            n_offset = offset_number(max(-self.x_left, self.x_right))
        return str(self.nlanes) + offset_code + n_offset
        
    def __repr__(self):
        return self.__str__()
    

class BaseRoad(Segment):
    def __init__(self, units, x0):
        #Construct object
        super(BaseRoad, self).__init__(units, units, x_left=[x0, x0])
        self.units = self.start
        self.x = self.x_start
    
    #Overrides decompose and __str__ methods
    def decompose(self):
        return [Carriageway(l[1] - l[0], self.x_start[l[0]]) \
                for l in Segment.get_lane_blocks(self.start, self.first_lane)]
    
    def __str__(self):
        return "CSUR:" + "".join([str(x) for x in self.decompose()])


class TwoWay(Segment):
    def __init__(self, base_road, half=False):
        # We don't want maek two-way shift segments 
        if isinstance(base_road, Shift):
            raise ValueError("Shift segments are one-way only")
        d_start = base_road.start.copy()
        d_end = base_road.end.copy()
        
        # Create a wide median if necessary, using multiple median units
        if d_start[0] != Segment.MEDIAN:
            d_start.pop(0)
            d_end.pop(0)
            n_median = int(base_road.x_start[1] // Segment.widths[Segment.MEDIAN])
            d_start = [Segment.MEDIAN] * n_median + d_start
            d_end = [Segment.MEDIAN] * n_median + d_end
            
        
        d_start = d_start[::-1] + d_start
        d_end = d_end[::-1] + d_end
        dx_left = [-base_road.x_start[-1], -base_road.x_end[-1]]
        
        self.base_type = str(base_road).split(':')[0]
        super(TwoWay, self).__init__(d_start, d_end, 
                                     x_left=dx_left, first_lane = len(d_start) // 2)
        
        
    def __str__(self):
        names = self.decompose()
        name_start = "".join(["%dD%s" % (2 * int(str(x)[0]), str(x)[1:]) for x in names[0]])
        name_end = "".join(["%dD%s" % (2 * int(str(x)[0]), str(x)[1:]) for x in names[1]])
        return self.base_type + ':' \
                + (name_start if name_start == name_end else name_start + '=' + name_end)

class Transition(Segment):
    def __str__(self):
        return "CSUR-T:" + super(Transition, self).__str__()   

class Ramp(Segment):
    def __str__(self):
        return "CSUR-R:" + super(Ramp, self).__str__()   

class Shift(Segment):
    def __str__(self):
        return "CSUR-S:" + super(Shift, self).__str__()  

class Access(Segment):
    
    def __init__(self, *args, name=None, **kwargs):
        super(Access, self).__init__(*args, **kwargs)
        self.name_override = name
    def __str__(self):
        names = self.decompose()
        if self.name_override:
            return "CSUR-A:" + self.name_override + ">"+ str(names[1][1])   
        else:
            return "CSUR-A:" + str(names[0][0]) + ">"+ str(names[1][1])    

class CSURFactory():
    roadside = {'g': [Segment.MEDIAN, Segment.BIKE, Segment.CURB],
                'e': [Segment.BARRIER]
               }
    road_in = {'g': Segment.CURB, 'e': Segment.BARRIER}
    
    def get_units(mode, lane_left, *blocks, n_median=1, prepend_median=True):
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
        elif self.roadtype == 'a':
            self.get = self.get_access
        
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

    def get_ramp(self, lane_lefts, n_lanes, n_medians=[1, 1]):
        start, x0_start = CSURFactory.get_units(self.mode, 
                                                lane_lefts[0], n_lanes[0], n_median=n_medians[0])
        end, x0_end = CSURFactory.get_units(self.mode, 
                                            lane_lefts[1], n_lanes[1], n_median=n_medians[1])
        p = 0
        while p < len(start):
            while p < len(start) and start[p] == end[p]:
                p += 1
            if p < len(start):
                if start[p] == Segment.MEDIAN:
                    end.insert(p, Segment.EMPTY)
                elif end[p] == Segment.MEDIAN:
                    start.insert(p, Segment.EMPTY)
                p += 1
            if len(start) + 1 == len(end):
                start.insert(-len(CSURFactory.roadside[self.mode]), Segment.EMPTY)
            if len(end) + 1 == len(start):
                end.insert(-len(CSURFactory.roadside[self.mode]), Segment.EMPTY)
                
        return Ramp(start, end, [x0_start, x0_end])
    
    def get_shift(self, lane_lefts, *blocks):
        units, x0 = CSURFactory.get_units(self.mode, lane_lefts[0], *blocks)
        return Shift(units, units, [x0, x0 - lane_lefts[0] + lane_lefts[1]])

    def get_access(self, lane_left, gnd_road, i_a, n_a, name_override=None):
        units, x0 = CSURFactory.get_units(self.mode, lane_left, gnd_road)
        start = units.copy()
        end = units.copy()
        p = 0
        while (start[p] != Segment.LANE):
            p += 1
            
        # locate (i_a)th lane
        p += i_a - 1
        # Replace (n_a + 1) lanes with n_a lanes and 2 medians
        if p + n_a + 1> gnd_road:
            raise ValueError("Not enough lanes to build access road")
        end = end[:p] + [Segment.MEDIAN, Segment.EMPTY] + n_a * [Segment.LANE] \
                + [Segment.MEDIAN] + end[p + n_a + 1:]
        # Add empty units to instruct segment constructor
        start = start[:p] + [Segment.EMPTY] + start[p:p+n_a+1] + [Segment.EMPTY] + start[p + n_a + 1:]
        return Access(start, end, x_left=[x0, x0], name=name_override)
        