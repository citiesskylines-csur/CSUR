import json
from itertools import product
from copy import deepcopy
from core.assets import Asset, BaseAsset, TwoWayAsset, reverse
from core.csur import offset_x, StandardWidth as SW


DEFAULT_MODE = 'g'

N_MEDIAN = 2
WIDE_SPLIT_MIN = 8

N_SHIFT_MAX = 1.5
DN_TRANS = 1
DN_RAMP = 1

EXPRESS_LMAX = 3.5
ALLOWED_UTURN = [3,4,5]

# decorator to check only base roads are passed to the function
def check_base_road(func): 
    def wrapper(seg1, seg2, *args, **kwargs): 
        for arg in [seg1, seg2]:
            if arg.roadtype != 'b':
                raise ValueError("Connection modules should be made from base roads")
        if seg1 == seg2:
            raise ValueError("Two ends connected should be different: %s, %s" % (seg1, seg2))
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
                raise ValueError("Invalid shift increment! %s=%s" % (start, end))
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
            

def find_access(nlane, base, name=None, codes=['3', '3P', '4P', '5P', '6P']):
    access_roads = []
    nlane_g = base.get_blocks()[0].nlanes
    x0 = base.x0()
    offsets = [offset_x(code) for code in codes]
    for i_a in range(2, nlane_g - nlane):
        if x0 + (i_a + nlane - 1) * SW.LANE + SW.MEDIAN in offsets:
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
    ADD_LEFT = 1


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
        # add centered one-way nC modules with n<max_undivided
        if self.ADD_LEFT:
            for i in range(1, 3):
                self.base[i - 1].insert(0, BaseAsset(-SW.LANE * i / 2, i))
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
                if abs(roads[j].x0() - roads[j - 1].x0()) <= N_SHIFT_MAX * SW.LANE:
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
        for i in range(2, self.max_lane):
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
        for x in flatten(self.base[3:]):
            access.extend(find_access(1, x, codes=self.codes[0]))
            access.extend(find_access(2, x, codes=self.codes[1]))
        access.extend([reverse(a) for a in access])
        self.ramp.extend(access)

    def _find_twoway(self):

        for i in range(3, self.MAX_UNDIVIDED + 1):
            self.base[i - 1].insert(0, BaseAsset(-SW.LANE * i / 2, i))

        # first resolve undivided base segments,
        # remove all but 1R0P
        undivided_base = []
        for l in self.base:
            for i, r in enumerate(l.copy()[::-1]):
                if r.is_undivided():
                    if r.nl() > 1:
                        l.pop(len(l) - i - 1)
                    undivided_base.append(r)
       
        # remove all local-express undivided segments
        # we can always build them using two base segments
        for l in self.comp:
            ntot = len(l.copy())
            for i, r in enumerate(l.copy()[::-1]):
                if r.is_undivided() and r.get_blocks()[0].nlanes > 1:
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
            if r.x0() > 0 and (r.nl() > 1 or r.x0() <= self.MAX_TWOWAY_MEDIAN * SW.LANE):
                self.twoway.append(TwoWayAsset(r, r)) 
        
        # make comp segments with more than 4 lanes two-way
        for r in flatten([[x for x in self.comp[2] if x.x0() == 0]] + self.comp[3:]):
            if 0 <= r.x0() <= self.MAX_TWOWAY_MEDIAN * SW.LANE and (r.x0() == 0 or r.get_blocks()[0].nlanes > 1):
                self.twoway.append(TwoWayAsset(r, r))

        # find all undivided interface segments
        # need to account for double counting
        undivided_interface = []
        undivided_interface_sorted = []
        for i in range(len(self.shift) - 1, -1, -1):
            if self.shift[i].is_undivided():
                r = self.shift.pop(i)
                undivided_interface.append(r)
                if 0 <= r.xleft[0] < r.xleft[1]:  
                    undivided_interface_sorted.append(r)
        for i in range(len(self.trans) - 1, -1, -1):
            if self.trans[i].is_undivided():
                r = self.trans.pop(i)
                undivided_interface.append(r)
                if min(r.xleft) >= 0 and r.ntot_start() < r.ntot_end():
                    undivided_interface_sorted.append(r)
        for i in range(len(self.ramp) - 1, -1, -1):
            if self.ramp[i].is_undivided():
                r = self.ramp.pop(i)
                if self.ramp[i].nblock() < 5:
                    undivided_interface.append(r)
                    if min(r.xleft) >= 0 and  r not in undivided_interface_sorted and reverse(r) not in undivided_interface_sorted:
                        undivided_interface_sorted.append(reverse(r))
        for r in undivided_interface_sorted:
            self.twoway.append(TwoWayAsset(r, r))
        
        # add 1R0P-associated left exist assets back
        for r in undivided_interface:
            if (str(r._blocks[0][0]) == '1R0P' and r.xleft[1] != 0)\
                or (str(r._blocks[1][0]) == '1R0P' and r.xleft[0] != 0):
                if r.roadtype == 's':
                    self.shift.append(r)
                elif r.roadtype == 't':
                    self.trans.append(r)
                else:
                    self.ramp.append(r)
        
        #print(undivided_interface, len(undivided_interface))
        #print(undivided_interface_sorted)

        if self.ASYM_SLIPLANE:
            for r1, r2 in product(undivided_base, undivided_interface):
                if r2.always_undivided():
                    r_t = TwoWayAsset(r1, r2)
                    if abs(r_t.asym()[0]) + abs(r_t.asym()[1]) <= 1 and r2.nblock() < 4:
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
        for i in range(1, self.MAX_UNDIVIDED):
            l = BaseAsset(3 * SW.MEDIAN, i)
            r = BaseAsset(SW.MEDIAN, i + 1)
            self.twoway.append(TwoWayAsset(l, r))

        # asym (n-1)Rn-(n+1)Rn
        for i in range(1, self.MAX_UNDIVIDED):
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
        assets['comp'] = [x for x in flatten(self.comp[1:]) if not x.is_undivided()]
        assets['shift'] = self.shift  
        assets['trans'] = self.trans
        assets['ramp'] = [x for x in self.ramp if x.nblock() == 3]
        assets['ramp'] += [x for x in self.ramp if abs(len(x._blocks[0]) - len(x._blocks[1])) == 2]
        assets['ramp'] += [x for x in self.ramp if x.nblock() > 3 \
                             and abs(x.get_blocks()[0][0].nlanes - x.get_blocks()[1][0].nlanes) < 3\
                             and x.get_blocks()[0][1].x_left - x.get_blocks()[0][0].x_right == SW.MEDIAN]
        assets['twoway'] = self.twoway
        # test assets
        for v in assets.values():
            for r in v:
                try:
                    for m in ['g', 'e']:
                        r.get_model(m)
                except Exception as e:
                    print("Asset %s failed: %s" % (r, e))
        return assets

    def get_variants(self):
        if not self.built:
            raise Exception("Asset pack not built; use self.build() to build")
        variants = {}
        assets = self.get_assets()
        right = lambda x: max([block[-1].x_right for block in x.get_all_blocks()])
        # ground express, all single-carriageway roads <= 3.5L
        variants['express'] = [x for x in assets['base'] + assets['shift'] \
            + assets['trans'] + assets['ramp'] + assets['twoway'] if right(x) <= 5 * SW.LANE]
        # ground compact, all roads w/ traffic lights <= 3.5L
        variants['compact'] = [x for x in assets['base'] + assets['twoway'] 
                        if x.has_trafficlight() and right(x) <= 3.5 * SW.LANE]
        # uturn lanes, with median size 3L, 4L and 5L
        variants['uturn'] = []
        # BRT stations, comp segment with 2DC in the middle
        variants['brt'] = []
        for x in assets['twoway']:
            if str(x.left) == str(x.right) and x.right.x0() / SW.LANE in ALLOWED_UTURN:
                y = Asset(x.right.x0(), x.right.nl(), x.right.x0() - SW.LANE, x.right.nl() + 1)
                variants['uturn'].append(TwoWayAsset(y, y))
            if str(x.left) == str(x.right) and x.right.x0() == 0 and len(x.right.nlanes[0]) > 1:
                y = Asset(x.right.x0(), x.right.nlanes[0], medians=[2,2])
                variants['brt'].append(TwoWayAsset(y, y))
        return variants

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

'''
Assembles the asset dictionary into multiple packages.
R1: Roads & Streets Pack 
containing 
    one-way 1C--3C, two-way 2DC, 4DC, 4DR--10DR, 6DR2DR4P, 6DR4DR5P and 8DR2DR5P
dependency: None

R2: Roads & Streets Wide Pack
containing
    4C, 6DC, 2DR, 2DR2--8DR5, 4DR4DR4P, and 4DR6DR5P
dependency: R1

R3: Roads & Streets Extended Pack
containing
    all other symmetric two-way roads
dependency: R1, R2

R4: Roads & Streets Asymmetric Pack 
containing
    all asymmetric two-way roads
dependency: R1, R2

M1: Modular Interchanges Lite Pack 
containing
    base: 1R3--1R4P, 2R--2R5P, (3--5)R
    shift: 2R--2R5P
    trans: all between base, 4DR4P=6DR4P uturn, 4DR5P=6DR5P uturn
    ramp: 2R4P, 3R--5R, 3R2R5P<=>4R1R5P
dependency: R1

M2: Modular Interchanges Full Pack 
containing:
    all modules strictly below 5R, divided, not present in M1
dependency: M1

M3: Modular Interchanges Slim Pack 
containing:
    all modules strictly undivided
dependency: M1

M4: Modular Interchanges 5-Lane Extension 
containing:
    all modules strictly below 6R and not present in M1--M3,
    except for 2 to 3 ramps
dependency: R2, M2

M5: Modular Interchanges 6-Lane Extension
containing:
    all modules below or equal 7R and not present in M1--M4,
    except for 2 to 3 ramps,
dependency: R3, M2

M6: Modular Interchanges Multi-Fork Pack  
containing:
    all 2 to 3 ramp modules
dependency: M5

B: BRT Pack / 快速公交包
containing all roads associated with BRT.
dependency: None


'''
def get_packages(assets, variants):
    packages = {}
    for i in range(1, 5):
        packages['R%i'%i] = []
    for i in range(1, 7):
        packages['M%i'%i] = []
    packages['B'] = []

    assets = deepcopy(assets)
    variants = deepcopy(variants)

    x_max = lambda a : max(a._blocks[0][-1].x_right, a._blocks[1][-1].x_right)
    
    for a in assets['base'] + assets['comp']:
        if a.x0() + a.x1() == 0:
            packages['R1' if a.nl() < 4 else 'R2'].append(a)
        elif a.nl() == 1 and 2.5 * SW.LANE <= a.x0() <= 4.5 * SW.LANE:
            packages['M1'].append(a)
        elif a.nblock() == 2 and a.nl() == 2 and a.x0() <= 4.5 * SW.LANE:
            packages['M1'].append(a)
        elif a.nblock() == 2 and 3 <= a.nl() <= 5 and a.x0() == 0.5 * SW.LANE:
            packages['M1'].append(a)
        elif a.x1() < 5.5 * SW.LANE:
            packages['M2'].append(a)
        elif a.x1() <= 6.5 * SW.LANE:
            packages['M4'].append(a)
        else:
            packages['M5'].append(a)

    for a in assets['shift']:
        if a.nl_min() == 2 and x_max(a) <= 6 * SW.LANE:
            packages['M1'].append(a)
        elif x_max(a) < 5.5 * SW.LANE:
            packages['M2'].append(a)
        elif x_max(a) <= 6.5 * SW.LANE:
            packages['M4'].append(a)
        else:
            packages['M5'].append(a)

    for a in assets['trans']:
        if BaseAsset(a.xleft[0], *a.nlanes[0]) in packages['M1'] \
             and BaseAsset(a.xleft[1], *a.nlanes[1]) in packages['M1']:
            packages['M1'].append(a)
        elif x_max(a) < 5.5 * SW.LANE:
            packages['M2'].append(a)
        elif x_max(a) <= 6.5 * SW.LANE:
            packages['M4'].append(a)
        else:
            packages['M5'].append(a)

    anchors = ['2R4P', '3R', '4R', '5R']
    anchors2 = ['1R31R4P', '2R1R3P', '2R2R4P', '3R1R4P', '3R2R5P']
    for a in assets['ramp']:
        if a.nblock() < 5:
            if (str(a).split('=')[0] in anchors and str(a).split('=')[1] in anchors2) \
                or (str(a).split('=')[1] in anchors and str(a).split('=')[0] in anchors2) \
                or str(a) in ['3R2R5P=4R1R5P', '4R1R5P=3R2R5P']:
                packages['M1'].append(a)
            elif x_max(a) < 5.5 * SW.LANE:
                packages['M2'].append(a)
            elif x_max(a) <= 6.5 * SW.LANE:
                packages['M4'].append(a)
            else:
                packages['M5'].append(a)
        else:
            packages['M6'].append(a)


    for a in assets['twoway'] + variants['uturn'] + variants['brt']:
        if a.roadtype == 'b' and a.is_symmetric():
            if a.is_undivided():
                if a.nblock() == 4:
                    if a.nl_min() <= 2:
                        packages['R1'].append(a)
                    elif a.nl_min() == 3:
                        packages['R2'].append(a)
                    else:
                        packages['R3'].append(a)
                else:
                    packages['B'].append(a)
            else:
                # a.nblock() == 4 for regular two-way, 8 for local-express
                if a.nblock() == 4:
                    if a.get_dim()[0] <= 12 * SW.LANE:
                        if a.nl_min() >= 2 and a.n_median_min() == 2:
                            packages['R1'].append(a)
                        else:
                            packages['R2'].append(a)
                    else:
                        packages['R3'].append(a)
                else:
                    if a.get_dim()[0] <= 12 * SW.LANE:
                        if a.n_median_min() == 2 and a.right.nlanes[0][0] > a.right.nlanes[0][1]:
                            packages['R1'].append(a)
                        elif a.n_median_min() == 0:
                            packages['B'].append(a)
                        else:
                            packages['R2'].append(a)
                    else:
                        packages['R3'].append(a)
        elif a.roadtype == 'b' and not a.is_symmetric():
            if a.has_trafficlight():
                packages['R4'].append(a)
            else:
                packages['M3'].append(a)
        else:
            if a.left.always_undivided() and a.right.always_undivided():
                packages['M3'].append(a)
            elif str(a) in ['4DR5P=6DR5P', '4DR4P=6DR4P']:
                packages['M1'].append(a)
            elif max(a.get_dim()) < 11 * SW.LANE:
                packages['M2'].append(a)
            elif max(a.get_dim()) <= 13 * SW.LANE:
                packages['M4'].append(a)
            else:
                packages['M5'].append(a)
    return packages




        

        