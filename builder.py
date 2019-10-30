import json
from assets import Asset, BaseAsset, TwoWayAsset, reverse
from csur import offset_x
from csur import StandardWidth as SW
from itertools import product

DEFAULT_MODE = 'g'

N_MEDIAN = 2
WIDE_SPLIT_MIN = 6

N_SHIFT_MAX = 2.0
DN_TRANS = 1
DN_RAMP = 1

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
def combine(express, local):
    n_median = (local.x0() - express.x1()) / SW.MEDIAN
    nl_comb = [x.nlanes for x in express.get_blocks() + local.get_blocks()]
    if int(n_median) == n_median and n_median > 0:
        return BaseAsset(express.x0(), *nl_comb, median=int(n_median))
    else:
        raise ValueError("Invalid parallel combination!", express, local)
    
@check_base_road
def connect(start, end):
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
            if abs(n0 - n1) > DN_TRANS:
                raise ValueError("Invalid transition increment! %s=%s" % (start, end))
            elif x0_l != x1_l and x0_r != x1_r:
                raise ValueError("Invalid transition alignment! %s=%s" % (start, end))
            return Asset(x0_l, n0, x1_l, n1)
    else:
        # Look for ramp
        n0 = [c.nlanes for c in start.get_blocks()]
        n1 = [c.nlanes for c in end.get_blocks()]
        if abs(sum(n0) - sum(n1)) > DN_RAMP or (x0_l != x1_l and x0_r != x1_r):
            raise ValueError("Invalid ramp combination! %s=%s" % (start, end))
        n_medians = [1, 1]
        if len(n0) > 1:
            n_medians[0] = int((start.get_blocks()[1].x_left - start.get_blocks()[0].x_right) // SW.MEDIAN)
        if len(n1) > 1:
            n_medians[1] = int((end.get_blocks()[1].x_left - end.get_blocks()[0].x_right) // SW.MEDIAN)
        return Asset(x0_l, n0, x1_l, n1, medians=n_medians)
            

def find_access(nlane, base, name=None, codes=['5', '5P', '6P', '7P', '8P']):
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
    # integer parameter
    N_MEDIAN = 1
    WIDE_SPLIT_MIN = 6
    DN_TRANS = 1
    MAX_UNDIVIDED = 4
    MAX_TWOWAY_MEDIAN = 1.5
    # boolean parameters
    USE_DN_RAMP = 0
    ASYM_SLIPLANE = 1


    def __init__(self, base_init, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k.upper()):
                setattr(self, k.upper(), v)
            else:
                raise ValueError("Invalid settings item: %s" % k)
        
        self.max_lane = len(base_init)
        self.codes = base_init
        self.base = [find_base(i + 1, codes=base_init[i], mode=self.MODE) for i in range(self.max_lane)]
        for i in range(1, self.MAX_UNDIVIDED + 1):
            self.base[i - 1].insert(0, BaseAsset(0, i))
        self.built = False
        self.comp = []
        self.triplex = []
        self.shift = []
        self.trans = []
        self.ramp = []
        self.twoway = []
        
    def load_file(self, file):
        with open(file, 'r') as f:
            settings = json.load(f)
            for k, v in settings.items():
                if hasattr(self, k.upper()):
                    setattr(self, k.upper(), v)
                else:
                    raise ValueError("Invalid settings item: %s" % k)

    def _find_comp(self):
        self.comp = [[] for _ in range(self.max_lane)]
        pairs = []
        for p, q in product(flatten(self.base), repeat=2):
            sep = (q.x0() - p.x1()) / SW.MEDIAN
            if sep == 1.0 or (p.nl() + q.nl() >= self.WIDE_SPLIT_MIN and sep == int(sep) and sep > 0 and sep <= self.N_MEDIAN):
                pairs.append([p, q])
        for p in pairs:
            interface_piece = combine(p[0], p[1])
            #interface_piece.set_prev(p[0]).set_next(p[1])
            if interface_piece.nl() <= self.max_lane:
                self.comp[interface_piece.nl() - 1].append(interface_piece)
        
        #create triplex segments
        self.triplex = [[] for _ in range(self.max_lane)]
        pairs = []
        for p, q in product(flatten(self.base), flatten(self.comp)):
            sep = (q.x0() - p.x1()) / SW.MEDIAN
            if sep == 1.0 or (p.nl() + q.nl() >= self.WIDE_SPLIT_MIN and sep == int(sep) and sep > 0 and sep <= self.N_MEDIAN):
                pairs.append([p, q])
        for p in pairs:
            interface_piece = combine(p[0], p[1])
            #interface_piece.set_prev(p[0]).set_next(p[1])
            if interface_piece.nl() <= self.max_lane:
                self.triplex[interface_piece.nl() - 1].append(interface_piece)
    
    def _find_shift(self):
        pairs = []
        for roads in self.base:
            for j in range(1, len(roads)):
                if roads[j].x0() - roads[j - 1].x0() <= N_SHIFT_MAX * SW.LANE:
                    pairs.append((roads[j - 1], roads[j]))
                    pairs.append((roads[j], roads[j - 1]))
        for p in pairs:
            interface_piece = connect(p[0], p[1])
            self.shift.append(interface_piece)

    def _find_trans(self):
        pairs = []
        # iterate from fewer to more lanes
        for i in range(self.max_lane):
            for j in range(i + 1, min(i + self.DN_TRANS + 1, self.max_lane)):
                p_cur = [p for p in product(self.base[i], self.base[j]) \
                        if (p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1())]
                p_cur += [(p[1], p[0]) for p in p_cur]
                pairs.extend(p_cur)
        for p in pairs:
            interface_piece = connect(p[0], p[1])
            #interface_piece.set_prev(p[0]).set_next(p[1])
            self.trans.append(interface_piece)

    def _find_ramp(self):
        ramp = []
        pairs = []
        # 1 to 2 ramp
        for i in range(1, self.max_lane):
            p_cur = [p for p in product(self.base[i], self.comp[i]) \
                        if (p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1())]
            p_cur += [(p[1], p[0]) for p in p_cur]
            pairs.extend(p_cur)
        
        # 1 to 2 ramp with transition
        if self.USE_DN_RAMP:
            for i in range(self.max_lane - 1):
                p_cur = [p for p in product(self.base[i], self.comp[i + 1]) \
                            if (p[0].x0() == p[1].x0()) #or p[0].x1() == p[1].x1()) \
                            and (p[1].x1() - p[0].x1() + p[1].x0() - p[0].x0() <= SW.LANE + SW.MEDIAN) \
                        ]
                p_cur += [(p[1], p[0]) for p in p_cur]
                pairs.extend(p_cur)
        # 2 to 2 ramp
        # no need to reverse because of exchange symmetry    
        for i in range(3, self.max_lane):
            p_cur = [p for p in product(self.comp[i], self.comp[i]) \
                        if (p[0].x0() == p[1].x0() and p[0].x1() == p[1].x1() and p[1] is not p[0])]
            pairs.extend(p_cur)

        # 2 to 3 ramp, starts with 4 lanes
        # another constraint: number of lanes should be differents at inner and outer ends
        for i in range(4, self.max_lane):
            p_cur = [p for p in product(self.comp[i], self.triplex[i]) \
                        if ((p[0].x0() == p[1].x0() or p[0].x1() == p[1].x1()) \
                            and p[0].get_blocks()[0].nlanes != p[1].get_blocks()[0].nlanes \
                            and p[0].get_blocks()[-1].nlanes != p[1].get_blocks()[-1].nlanes \
                            )]
            p_cur += [(p[1], p[0]) for p in p_cur]            
            pairs.extend(p_cur)

        for p in pairs:
            interface_piece = connect(p[0], p[1])
            #interface_piece.set_prev(p[0]).set_next(p[1])
            self.ramp.append(interface_piece)
        
        # 1 to 3 ramp
        access = []
        for x in flatten(self.base[4:]):
            access.extend(find_access(1, x, codes=self.codes[0])) 
            access.extend(find_access(2, x, codes=self.codes[1]))
        access.extend([reverse(a) for a in access])
        self.ramp.extend(access)

    def _find_twoway(self):
        # first resolve undivided base segments
        undivided_base = []
        for l in self.base:
            for i, r in enumerate(l.copy()[::-1]):
                if r.is_undivided():
                    l.pop(len(l) - i - 1)
                    undivided_base.append(r)
        # remove all local-express undivided segments
        # we can always build them using two base segments
        # TODO: still keep the BRT-related segments with 2DC inside
        for l in self.comp:
            ntot = len(l.copy())
            for i, r in enumerate(l.copy()[::-1]):
                if r.is_undivided():
                    l.pop(ntot - i - 1)
        for r in undivided_base:
            self.twoway.append(TwoWayAsset(r, r))
        if self.ASYM_SLIPLANE:
            for r1, r2 in product(undivided_base, repeat=2):
                if r2.nl() - r1.nl() == 1:
                    self.twoway.append(TwoWayAsset(r1, r2))

        for r in flatten(self.base):
            # make base segments less than 1.5u median two-way
            # make other > 2 lanes also twoway:
            if r.nl() > 1 or r.x0() <= self.MAX_TWOWAY_MEDIAN * SW.LANE:
                self.twoway.append(TwoWayAsset(r, r))
            
            
        # make comp segments with more than 4 lanes two-way
        for r in flatten(self.comp[3:]):
            if r.x0() <= self.MAX_TWOWAY_MEDIAN * SW.LANE:
                self.twoway.append(TwoWayAsset(r, r))

        # find all undivided interface segments
        # need to account for double counting
        undivided_interface = []
        for i in range(len(self.shift) - 1, -1, -1):
            if self.shift[i].is_undivided():
                r = self.shift.pop(i)
                if r.xleft[0] < r.xleft[1]:
                    undivided_interface.append(r)
        for i in range(len(self.trans) - 1, -1, -1):
            if self.trans[i].is_undivided():
                r = self.trans.pop(i)
                if r.ntot_start() < r.ntot_end():
                    undivided_interface.append(r)
        for i in range(len(self.ramp) - 1, -1, -1):
            if self.ramp[i].is_undivided():
                r = self.ramp.pop(i)
                if len(r._blocks[0]) == 1:
                    undivided_interface.append(r)
        for r in undivided_interface:
            self.twoway.append(TwoWayAsset(r, r))

        if self.ASYM_SLIPLANE:
            for r1, r2 in product(undivided_base, undivided_interface):
                if r2.always_undivided():
                    r_t = TwoWayAsset(r1, r2)
                    if abs(r_t.asym()[0]) + abs(r_t.asym()[1]) <= 1:
                        if sum(r_t.asym()) > 0:
                            self.twoway.append(r_t)
                        else:
                            self.twoway.append(TwoWayAsset(r2, r1))

    def _find_asym(self):
        # asym (2n+1)DC
        for i in range(1, self.MAX_UNDIVIDED):
            l = BaseAsset(SW.MEDIAN, i)
            r = BaseAsset(-SW.MEDIAN, i + 1)
            self.twoway.append(TwoWayAsset(l, r))
        # asym (n-1)Rn-nR
        for i in range(1, self.max_lane):
            l = BaseAsset(3 * SW.MEDIAN, i)
            r = BaseAsset(SW.MEDIAN, i + 1)
            self.twoway.append(TwoWayAsset(l, r))

        # asym (n-1)Rn-(n+1)Rn
        for i in range(1, self.max_lane):
            l = BaseAsset(3 * SW.MEDIAN, i)
            r = BaseAsset(-SW.MEDIAN, i + 2)
            self.twoway.append(TwoWayAsset(l, r))

    def build(self, twoway=True):
        self._find_comp()
        self._find_shift()
        self._find_trans()
        self._find_ramp()
        if twoway:
            self._find_twoway()
            self._find_asym()
        self.built = True
        return self

    def get_assets(self):
        if not self.built:
            raise Exception("Asset pack not built; use self.build() to build")
        assets = {}
        assets['base'] = flatten(self.base)
        assets['comp'] = flatten(self.comp[3:])
        assets['shift'] = self.shift   
        assets['trans'] = self.trans
        assets['ramp'] = [x for x in self.ramp if x.nblock() == 3]
        assets['ramp'] += [x for x in self.ramp if abs(len(x._blocks[0]) - len(x._blocks[1])) == 2]
        assets['ramp'] += [x for x in self.ramp if x.nblock() > 3 \
                             and abs(x.get_blocks()[0][0].nlanes - x.get_blocks()[1][0].nlanes) < 3\
                             and x.get_blocks()[0][1].x_left - x.get_blocks()[0][0].x_right == SW.MEDIAN]
        assets['twoway'] = self.twoway
        return assets

    def get_dependency(self, new_asset):
        if not self.built:
            raise Exception("Cannot resolve dependency: asset pack not built!")
        if new_asset.roadtype != 'b':
            raise ValueError("Dependency can only be calculated for base asset!")
        # TODO: check for existing assets in the pack
        dependencies = []

        pairs = []
        # find shift segments to add
        for road in self.base[new_asset.nl() - 1]:
            if abs(road.x0() - new_asset.x0()) <= N_SHIFT_MAX * SW.LANE:
                pairs.append((road, new_asset))
                pairs.append((new_asset, road))
    
        # find transition segments to add
        # also add composite segments to fork
        candidates = self.comp[new_asset.nl() - 1].copy()
        if new_asset.nl() > 1:
            candidates += self.base[new_asset.nl() - 2]
        if new_asset.nl() < len(self.base):
            candidates += self.base[new_asset.nl()]
        for road in candidates:
            if road.x0() == new_asset.x0() or road.x1() == new_asset.x1():
                pairs.append((road, new_asset))
                pairs.append((new_asset, road))
        # find composite segments to add
        left_neighbors = []
        left_temp = []
        right_neighbors = []
        forks = []
        for road in flatten(self.base):
            lspace = new_asset.x0() - road.x1()
            rspace = road.x0() - new_asset.x1()
            nlane = road.nl() + new_asset.nl()
            if lspace == SW.MEDIAN or \
                (nlane >= WIDE_SPLIT_MIN and lspace > 0 and lspace / SW.MEDIAN == int(lspace / SW.MEDIAN)):
                left_temp.append(road)
                left_neighbors.append(combine(road, new_asset))
            if rspace == SW.MEDIAN or \
                (nlane >= WIDE_SPLIT_MIN and rspace > 0 and rspace / SW.MEDIAN == int(rspace / SW.MEDIAN)):
                right_neighbors.append(combine(new_asset, road))   
        # add duplex pairs
        for duplex in left_neighbors + right_neighbors:
            if duplex.nl() <= self.max_lane:
                self.comp[duplex.nl() - 1].append(duplex)
                if duplex.nl() > 3:
                    dependencies.append(duplex)
                for road in self.base[duplex.nl() - 1]:
                    if road.x0() == duplex.x0() or road.x1() == duplex.x1():
                        pairs.append((road, duplex))
                        pairs.append((duplex, road))
        # add triplex pairs
        for triplex in [combine(left, duplex_r) \
                    for left, duplex_r in product(left_temp, right_neighbors)]:
            if triplex.nl() <= self.max_lane:
                self.triplex[triplex.nl() - 1].append(triplex)
                for comp in self.comp[triplex.nl() - 1]:
                    if road.x0() == triplex.x0() or road.x1() == triplex.x1():
                        pairs.append((comp, triplex))
                        pairs.append((triplex, comp))
        #print(pairs, left_neighbors, right_neighbors)
        for p in pairs:
            interface_piece = connect(p[0], p[1])
            dependencies.append(interface_piece)
        # update the added modules to the asset pack
        self.base[new_asset.nl() - 1].append(new_asset)

        for road in dependencies:
            if road.roadtype == 's':
                self.shift.append(road)
            elif road.roadtype == 't':
                self.trans.append(road)
            elif road.roadtype == 'r':
                self.ramp.append(road)
        return dependencies
        

        