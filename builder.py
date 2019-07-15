import json
from csur import Carriageway, offset_x, get_name, combine_name, twoway_reduced_name, CSURFactory
from csur import StandardWidth as SW
from itertools import product

DEFAULT_MODE = 'g'

N_MEDIAN = 2
WIDE_SPLIT_MIN = 6

N_SHIFT_MAX = 2.0
DN_TRANS = 1
DN_RAMP = 1

class Asset():
    def __init__(self, x0_start, nlanes_start, x0_end=None, nlanes_end=None, medians=None):
        if type(nlanes_start) == int:
            nlanes_start = [nlanes_start]
        if type(nlanes_end) == int:
            nlanes_end = [nlanes_end]
        self.xleft = [x0_start, x0_end or x0_start]
        self.nlanes = [nlanes_start, nlanes_end or nlanes_start]
        self.medians = medians or [1, 1] 
        self._infer_roadtype()
        self._infer_blocks()
        self.pred = []
        self.succ = []
        self.left = []
        self.right = []

    def _infer_roadtype(self):
        if self.xleft[0] == self.xleft[1] and self.nlanes[0] == self.nlanes[1] and self.medians[0] == self.medians[1]:
            self.roadtype = 'b'
        elif len(self.nlanes[0]) == 1 and len(self.nlanes[1]) == 1:
            if self.nlanes[0] == self.nlanes[1]:
                self.roadtype = 's'
            else:
                self.roadtype = 't'
        else:
            self.roadtype = 'r'

    def _infer_blocks(self):
        self._blocks = [[], []]
        for i in range(2):
            x0 = self.xleft[i]
            for n in self.nlanes[i]:
                self._blocks[i].append(Carriageway(n, x0))
                x0 += n * SW.LANE + self.medians[i] * SW.MEDIAN
    
    def __eq__(self, other):
        return self.xleft == other.xleft and self.nlanes == other.nlanes and self.medians == other.medians
    
    def __str__(self):
        return combine_name(get_name(self._blocks))
    
    def __repr__(self):
        return str(self)
    
    def nl(self):
        return sum(x.nlanes for x in self._blocks[0])

    def get_blocks(self):
        return self._blocks

    def get_model(self, mode='g'):
        fac = CSURFactory(mode=mode, roadtype=self.roadtype)
        if self.roadtype == 'b':
            return fac.get(self.xleft[0], *self.nlanes[0], n_median=self.medians[0])
        elif self.roadtype == 's':
            return fac.get(self.xleft, self.nlanes[0][0])
        elif self.roadtype == 't':
            return fac.get(self.xleft, [self.nlanes[0][0], self.nlanes[1][0]])
        elif self.roadtype == 'r':
            return fac.get(self.xleft, self.nlanes, n_medians=self.medians)

   
class BaseAsset(Asset):
    def __init__(self, x0_start, *nlanes_start, median=1):
        super().__init__(x0_start, nlanes_start, medians=[median, median])

    def get_blocks(self):
        return self._blocks[0]

    def x0(self):
        return self.get_blocks()[0].x_left
    
    def x1(self):
        return self.get_blocks()[-1].x_right
    
    def nl(self):
        return sum(x.nlanes for x in self._blocks[0])

class TwoWayAsset(Asset):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        names = [twoway_reduced_name(x, y) for x, y in zip(self.left._blocks, self.right._blocks)]
        return combine_name(names)

# decorator to check only base roads are passed to the function
def check_base_road(func): 
    def wrapper(seg1, seg2, *args, **kwargs): 
        for arg in [seg1, seg2]:
            if arg.roadtype != 'b':
                raise ValueError("Connection modules should be made from base roads")
        if seg1 == seg2:
            raise ValueError("Two ends connected should be different: %s, %s" % (seg1.obj, seg2.obj))
        return func(seg1, seg2, *args, **kwargs)  
    return wrapper

   
def find_base(nlane, codes=['5', '5P', '6P', '7P', '8P'], mode=DEFAULT_MODE):
    roads = []
    lefts = [offset_x(k) - nlane * SW.LANE for k in codes]
    for x in lefts:
        v = BaseAsset(x, nlane)
        roads.append(v)
    return roads


@check_base_road
def combine(express, local, mode=DEFAULT_MODE):
    n_median = (local.x0() - express.x1()) / SW.MEDIAN
    nl_comb = [x.nlanes for x in express.get_blocks() + local.get_blocks()]
    if int(n_median) == n_median and n_median > 0:
        return BaseAsset(express.x0(), *nl_comb, median=int(n_median))
    else:
        raise ValueError("Invalid parallel combination!", express, local)
    
@check_base_road
def connect(start, end, mode=DEFAULT_MODE):
    x0_l = start.x0()
    x1_l = end.x0()
    x0_r = start.get_blocks()[-1].x_right
    x1_r = end.get_blocks()[-1].x_right
    
    if len(start.get_blocks()) == 1 and len(end.get_blocks()) == 1:
        n0 = start.get_blocks()[0].nlanes
        n1 = end.get_blocks()[0].nlanes
        # Look for shift
        if n0 == n1:
            if abs(x0_l - x1_l) > N_SHIFT_MAX * SW.LANE:
                raise ValueError("Invalid shift increment! %d->%d" % (x0_l, x1_l))
            return Asset(x0_l, n0, x1_l, n0)
        # Look for transition
        else:
            if abs(n0 - n1) > DN_TRANS or (x0_l != x1_l and x0_r != x1_r):
                raise ValueError("Invalid transition combination!")
            return Asset(x0_l, n0, x1_l, n1)
    else:
        # Look for ramp
        n0 = [c.nlanes for c in start.get_blocks()]
        n1 = [c.nlanes for c in end.get_blocks()]
        if abs(sum(n0) - sum(n1)) > DN_RAMP or (x0_l != x1_l and x0_r != x1_r):
            raise ValueError("Invalid ramp combination!")
        n_medians = [1, 1]
        if len(n0) > 1:
            n_medians[0] = int((start.get_blocks()[1].x_left - start.get_blocks()[0].x_right) // SW.MEDIAN)
        if len(n1) > 1:
            n_medians[1] = int((end.get_blocks()[1].x_left - end.get_blocks()[0].x_right) // SW.MEDIAN)
        return Asset(x0_l, n0, x1_l, n1, medians=n_medians)
            

def find_access(nlane, base, mode=DEFAULT_MODE, name=None, codes=['5', '5P', '6P', '7P', '8P']):
    access_roads = []
    nlane_g = base.get_blocks()[0].nlanes
    x0 = base.x0()
    offsets = [offset_x(code) for code in codes]
    for i_a in range(2, nlane_g - nlane):
        if x0 + i_a * SW.LANE + SW.MEDIAN in offsets:
            access_roads.append(Asset(x0, nlane_g, x0, [i_a - 1, nlane, nlane_g - nlane - i_a]))
    return access_roads

flatten = lambda l: [item for sublist in l for item in sublist]

class Builder:

    MODE = 'g'
    N_MEDIAN = 2
    WIDE_SPLIT_MIN = 6
    DN_TRANS = 1
    DN_RAMP = 1
    MAX_UNDIVIDED = 4

    def __init__(self, base_init, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k.upper()):
                setattr(self, k.upper(), v)
            else:
                raise ValueError("Invalid settings item: %s" % k)
        
        max_lane = len(base_init)
        self.base = [find_base(i + 1, codes=base_init[i], mode=self.MODE) for i in range(max_lane)]
        
            

    def load_file(self, file):
        with open(file, 'r') as f:
            settings = json.load(f)
            for k, v in settings.items():
                if hasattr(self, k.upper()):
                    setattr(self, k.upper(), v)
                else:
                    raise ValueError("Invalid settings item: %s" % k)


def generate_all(max_lane, codes_all=['5', '5P', '6P', '7P', '8P'], setting=None):
    assets = {}
    if not setting:
        setting = {'trans_ramp': False,
                   'max_undivided': 4
                    }

    non_uniform_offset = isinstance(codes_all[0], list)

    # create base segments
    base = []
    for i in range(1, max_lane + 1):
        base.append([])
        codes = codes_all[i - 1] if non_uniform_offset else codes_all
        for x in find_base(i, codes=codes):
            base[-1].append(x)
    
    assets['base'] = flatten(base)
    # create shift segments
    shift = []
    for roads in base:
        # shift 1 index
        pairs = []
        for j in range(1, len(roads)):
            if roads[j].x0() - roads[j - 1].x0() <= N_SHIFT_MAX * SW.LANE:
                pairs.append((roads[j - 1], roads[j]))
                pairs.append((roads[j], roads[j - 1]))
        # shift 2 index
        '''
        for j in range(2, len(roads)):         
            if roads[j].x0() - roads[j - 2].x0() <= N_SHIFT_MAX * SW.LANE:
                pairs.append((roads[j - 2], roads[j]))
                pairs.append((roads[j], roads[j - 2]))
        '''
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
    for p, q in product(assets['base'], repeat=2):
        sep = (q.x0() - p.x1()) / SW.MEDIAN
        if sep == 1.0 or (p.nl() + q.nl() >= WIDE_SPLIT_MIN and sep == int(sep) and sep > 0 and sep <= N_MEDIAN):
            pairs.append([p, q])
    for p in pairs:
        v_f = combine(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        if v_f.nl() <= max_lane:
            comp[v_f.nl() - 1].append(v_f)
    
    #create triplex segments
    triplex = [[] for _ in range(max_lane)]
    pairs = []
    for p, q in product(assets['base'], flatten(comp)):
        sep = (q.x0() - p.x1()) / SW.MEDIAN
        if sep == 1.0 or (p.nl() + q.nl() >= WIDE_SPLIT_MIN and sep == int(sep) and sep > 0 and sep <= N_MEDIAN):
            pairs.append([p, q])
    for p in pairs:
        v_f = combine(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        if v_f.nl() <= max_lane:
            triplex[v_f.nl() - 1].append(v_f)
    
    # create ramps
    ramp = []
    pairs = []
    # 1 to 2 ramp
    for i in range(1, max_lane):
        p_cur = [p for p in product(base[i], comp[i]) \
                     if (p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1())]
        p_cur += [(p[1], p[0]) for p in p_cur]
        pairs.extend(p_cur)
    
    # 1 to 2 ramp with transition
    if setting['trans_ramp']:
        for i in range(max_lane - 1):
            p_cur = [p for p in product(base[i], comp[i + 1]) \
                        if (p[0].x0() == p[1].x0()) #or p[0].x1() == p[1].x1()) \
                        and (p[1].x1() - p[0].x1() + p[1].x0() - p[0].x0() <= SW.LANE + SW.MEDIAN) \
                    ]
            p_cur += [(p[1], p[0]) for p in p_cur]
            pairs.extend(p_cur)
    # 2 to 2 ramp
    # no need to reverse because of exchange symmetry    
    for i in range(3, max_lane):
        p_cur = [p for p in product(comp[i], comp[i]) \
                     if (p[0].x0() == p[1].x0() and p[0].x1() == p[1].x1() and p[1] is not p[0])]
        pairs.extend(p_cur)

    # 2 to 3 ramp, starts with 4 lanes
    # another constraint: number of lanes should be differents at inner and outer ends
    for i in range(4, max_lane):
        p_cur = [p for p in product(comp[i], triplex[i]) \
                     if ((p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1()) \
                         and p[0].get_blocks()[0].nlanes != p[1].get_blocks()[0].nlanes \
                         and p[0].get_blocks()[-1].nlanes != p[1].get_blocks()[-1].nlanes \
                         )]
        p_cur += [(p[1], p[0]) for p in p_cur]            
        pairs.extend(p_cur)

    for p in pairs:
        v_f = connect(p[0], p[1])
        #v_f.set_prev(p[0]).set_next(p[1])
        ramp.append(v_f)
    
    # create access roads
    access = []
    for x in flatten(base[4:]):
        codes = codes_all[0] if non_uniform_offset else codes_all
        access.extend(find_access(1, x, codes=codes))
        codes = codes_all[1] if non_uniform_offset else codes_all
        access.extend(find_access(2, x, codes=codes))
    '''
    # handle special modules
    if (max_lane >= 6):
        special_6r9 = BaseAsset(SpecialFactory("6R9").get())
        access.extend(find_access(1, special_6r9, name="6R9", codes=codes))
        access.extend(find_access(2, special_6r9, name="6R9", codes=codes))
        base[5].append(special_6r9)
    '''      
    # post processing
    # local-express combinations should be at least 4 lanes
    # and express lanes should be at least 2
    assets['comp'] = flatten(comp[3:])
    #assets['comp'] = [x for x in assets['comp'] if x.get_blocks()[0].nlanes >= 2]
    assets['shift'] = shift   
    assets['trans'] = trans
    #
    assets['ramp'] = [x for x in ramp \
                        if len(x.get_blocks()[0]) + len(x.get_blocks()[1]) < 4 \
                        or (abs(x.get_blocks()[0][0].nlanes - x.get_blocks()[1][0].nlanes) < 3 \
                        and x.get_blocks()[0][1].x_left - x.get_blocks()[0][0].x_right == SW.MEDIAN)
                     ]
    assets['access'] = access
    return assets