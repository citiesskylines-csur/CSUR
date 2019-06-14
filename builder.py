from csur import offset_x, CSURFactory
from csur import StandardWidth as SW
from itertools import product
import csur

DEFAULT_MODE = 'e'

N_MEDIAN = 2

N_MEDIAN_COMB = N_MEDIAN
N_MEDIAN_RAMP = N_MEDIAN

N_SHIFT = 1.5
DN_TRANS = 2
DN_RAMP = 1

class RoadAsset():
    def __init__(self, roadtype, road_obj):
        self.roadtype = roadtype
        self.blocks = road_obj.decompose()
        self.obj = road_obj
        self.prev = []
        self.next = []
        
    def draw(self):
        self.obj.draw()
    
    def __str__(self):
        return "Asset " + str(self.obj)
    
    def __repr__(self):
        return str(self)
    
    def set_prev(self, v_prev):
        self.prev = v_prev
        v_prev.next = self
        return self
        
    def set_next(self, v_next):
        self.next = v_next
        v_next.prev = self
        return self

class BaseAsset(RoadAsset):
    def __init__(self, obj):
        super(BaseAsset, self).__init__('b', obj)
    def x0(self):
        return self.blocks[0].x_left
    
    def x1(self):
        return self.blocks[-1].x_right
    
    def nl(self):
        return sum(x.nlanes for x in self.blocks)
    
# decorator to check only base roads are passed to the function
def check_base_road(func): 
    def wrapper(seg1, seg2, *args): 
        for arg in [seg1, seg2]:
            if not isinstance(arg.obj, csur.BaseRoad):
                raise ValueError("Connection modules should be made from base roads")
        if seg1.obj.x == seg2.obj.x:
            raise ValueError("Two ends connected should be different")
        return func(seg1, seg2, *args)  
    return wrapper

class Wide_6R(csur.BaseRoad):
    def __str__(self):
        return "CSUR:6R-Wide"
    
class SpecialFactory(CSURFactory):
    def __init__(self, name, mode='e'):
        if name == '6R9':
            self.get = self.get_6r_wide
        self.mode = mode
    
    def get_6r_wide(self):
        units, x0 = CSURFactory.get_units(self.mode, 3 * SW.MEDIAN, 6)
        return Wide_6R(units, x0)
    
def find_base(nlane, mode=DEFAULT_MODE, max_code=8):
    base = CSURFactory(mode=mode, roadtype='b')
    roads = []
    codes = ['5'] + ["%dP" % x for x in range(5, max_code + 1)]
    lefts = [offset_x(k) - nlane * SW.LANE for k in codes]
    if lefts[0] != SW.MEDIAN:
        lefts.insert(0, SW.MEDIAN)
    for x in lefts:
        if (x == SW.MEDIAN and nlane > 1) or x > 2 * SW.MEDIAN:
            v = BaseAsset(base.get(x, nlane))
            roads.append(v)
    return roads


@check_base_road
def combine(express, local, mode=DEFAULT_MODE):
    base = CSURFactory(mode=mode, roadtype='b')
    n_median = (local.x0() - express.blocks[0].x_right) / SW.MEDIAN
    if int(n_median) == n_median and n_median > 0 and n_median <= N_MEDIAN:
        return BaseAsset(base.get(express.x0(),
                                express.blocks[0].nlanes,
                                local.blocks[0].nlanes,
                                n_median=int(n_median)))
    else:
        raise ValueError("Invalid parallel combination!", express, local)
    #config = [express.config[0], express_r.]
    
@check_base_road
def connect(start, end, mode=DEFAULT_MODE):
    x0_l = start.x0()
    x1_l = end.x0()
    x0_r = start.blocks[-1].x_right
    x1_r = end.blocks[-1].x_right
    
    if len(start.blocks) == 1 and len(end.blocks) == 1:
        n0 = start.blocks[0].nlanes
        n1 = end.blocks[0].nlanes
        # Look for shift
        if n0 == n1:
            if abs(x0_l - x1_l) > N_SHIFT * SW.LANE:
                raise ValueError("Invalid shift increment! %d->%d" % (x0_l, x1_l))
            fac = CSURFactory(mode=mode, roadtype='s')
            return RoadAsset('s', fac.get([x0_l, x1_l], n0))
        # Look for transition
        else:
            if abs(n0 - n1) > DN_TRANS or (x0_l != x1_l and x0_r != x1_r):
                raise ValueError("Invalid transition combination!")
            fac = CSURFactory(mode=mode, roadtype='t')
            return RoadAsset('t', fac.get([x0_l, x1_l], [n0, n1], left=(x0_l!=x1_l)))
    else:
        # Look for ramp
        n0 = [c.nlanes for c in start.blocks]
        n1 = [c.nlanes for c in end.blocks]
        if abs(sum(n0) - sum(n1)) > DN_RAMP or (x0_l != x1_l and x0_r != x1_r):
            raise ValueError("Invalid ramp combination!")
        fac = CSURFactory(mode=mode, roadtype='r')
        n_medians = [1, 1]
        if len(n0) > 1:
            n_medians[0] = int((start.blocks[1].x_left - start.blocks[0].x_right) // SW.MEDIAN)
        if len(n1) > 1:
            n_medians[1] = int((end.blocks[1].x_left - end.blocks[0].x_right) // SW.MEDIAN)
        return RoadAsset('r', fac.get([x0_l, x1_l], [n0, n1], n_medians=n_medians))
            

def find_access(nlane, base, mode=DEFAULT_MODE, name=None, max_code=8):
    access_roads = []
    fac = CSURFactory(mode=mode, roadtype='a')
    nlane_g = base.blocks[0].nlanes
    x0 = base.x0()
    codes = ['5'] + ["%dP" % x for x in range(5, max_code + 1)]
    offsets = [offset_x(code) for code in codes]
    for i_a in range(1, nlane_g - nlane):
        if x0 + i_a * SW.LANE + SW.MEDIAN in offsets:
            access_roads.append(RoadAsset('a', 
                        fac.get(x0, nlane_g, i_a, nlane, name_override=name)))
    return access_roads


flatten = lambda l: [item for sublist in l for item in sublist]

def generate_all(max_lane, max_code=8, setting={}):
    assets = {}
    
    # create base segments
    base = []
    for i in range(1, max_lane + 1):
        base.append([])
        for x in find_base(i, max_code=max_code):
            base[-1].append(x)
    
    assets['base'] = flatten(base)
    
    # create shift segments
    shift = []
    for roads in base:
        # shift 1 index, should always within shift increment
        pairs = []
        for j in range(1, len(roads)):
            pairs.append((roads[j - 1], roads[j]))
            pairs.append((roads[j], roads[j - 1]))
        # shift 2 index, may out of allowable increment
        for j in range(2, len(roads)):         
            if roads[j].x0() - roads[j - 2].x0() <= N_SHIFT * SW.LANE:
                pairs.append((roads[j - 2], roads[j]))
                pairs.append((roads[j], roads[j - 2]))
        for p in pairs:
            v_f = connect(p[0], p[1])
            #v_f.set_prev(p[0]).set_next(p[1])
            shift.append(v_f)
            
    # create trans segments
    trans = []
    pairs = []
    # iterate from fewer to more lanes
    for i in range(max_lane):
        for j in range(i + 1, min(i + DN_TRANS + 1, max_lane)):
            p_cur = [p for p in product(base[i], base[j]) \
                     if (p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1())]
            p_cur += [(p[1], p[0]) for p in p_cur]
            pairs.extend(p_cur)
    for p in pairs:
        v_f = connect(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        trans.append(v_f)
        
    # create local-express segments
    comp = [[] for _ in range(max_lane)]
    pairs = []
    for i in range(len(assets['base'])):
        for j in range(len(assets['base'])):
            sep = (assets['base'][j].x0() - assets['base'][i].x1()) / SW.MEDIAN
            if sep > 0 and sep == int(sep) and sep <= N_MEDIAN:
                pairs.append([assets['base'][i], assets['base'][j]])
    for p in pairs:
        v_f = combine(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        if v_f.nl() <= max_lane:
            comp[v_f.nl() - 1].append(v_f)
    
    # create ramps
    ramp = []
    pairs = []
    for i in range(max_lane):
        p_cur = [p for p in product(base[i], comp[i]) \
                     if (p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1())]
        p_cur += [(p[1], p[0]) for p in p_cur]
        pairs.extend(p_cur)
        
    for p in pairs:
        v_f = connect(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        ramp.append(v_f)
    
    # create access roads
    access = []
    for x in flatten(base[4:]):
        access.extend(find_access(1, x, max_code=max_code))
        access.extend(find_access(2, x, max_code=max_code))
            
    # handle special modules
    if (max_lane >= 6):
        special_6r9 = BaseAsset(SpecialFactory("6R9").get())
        access.extend(find_access(1, special_6r9, name="6R9", max_code=max_code))
        access.extend(find_access(2, special_6r9, name="6R9", max_code=max_code))
        base[5].append(special_6r9)
            
    
    assets['comp'] = flatten(comp)
    assets['shift'] = shift   
    assets['trans'] = trans
    assets['ramp'] = ramp
    assets['access'] = access
    return assets