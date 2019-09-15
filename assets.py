
from csur import Segment, Carriageway, get_name, combine_name, twoway_reduced_name, CSURFactory, TwoWay
from csur import StandardWidth as SW


reverse = lambda a: Asset(a.xleft[1], a.nlanes[1], a.xleft[0], a.nlanes[0], a.medians)

class Asset():
    def __init__(self, x0_start, nlanes_start, x0_end=None, nlanes_end=None, medians=None):
        if type(nlanes_start) == int:
            nlanes_start = [nlanes_start]
        if type(nlanes_end) == int:
            nlanes_end = [nlanes_end]
        self.xleft = [x0_start, x0_start] if x0_end is None else [x0_start, x0_end]
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

    def nblock(self):
        return sum(len(x) for x in self._blocks)

    def ntot_start(self):
        return sum(self.nlanes[0])
    
    def ntot_end(self):
        return sum(self.nlanes[1])
    
    def nl(self):
        return self.ntot_start()

    def is_undivided(self):
        return self.xleft[0] == 0 or self.xleft[1] == 0

    def is_twoway(self):
        return False

    def has_sidewalk(self):
        return Segment.SIDEWALK in self.get_model('g').start

    def has_bikelane(self):
        return Segment.BIKE in self.get_model('g').start

    def is_roundabout(self):
        return Segment.WEAVE in self.get_model('g').start

    def has_trafficlight(self):
        return False

    def always_undivided(self):
        return self.xleft[0] == 0 and self.xleft[1] == 0

    def get_blocks(self):
        return self._blocks
    
    def get_all_blocks(self):
        return self._blocks

    def get_model(self, mode='g'):
        if mode[-1] == 'w':
            fac = CSURFactory(mode=mode[0], roadtype=self.roadtype)
        else:
            fac = CSURFactory(mode=mode, roadtype=self.roadtype)
        if self.roadtype == 'b':
            seg = fac.get(self.xleft[0], *self.nlanes[0], n_median=self.medians[0])
        elif self.roadtype == 's':
            seg = fac.get(self.xleft, self.nlanes[0], n_median=self.medians)
        elif self.roadtype == 't':
            seg = fac.get(self.xleft, [self.nlanes[0][0], self.nlanes[1][0]], left=self.xleft[0] != self.xleft[1])
        elif self.roadtype == 'r':
            seg = fac.get(self.xleft, self.nlanes, n_median=self.medians)
        if mode[-1] == 'w':
            if self.roadtype != 'b':
                raise ValueError("Weave segment is only available for base module!")
            seg.start = [Segment.WEAVE if i >= seg.units.index(Segment.LANE) \
                        and i + 1 + len(CSURFactory.roadside[mode[0]]) < len(seg.start)
                        and x == Segment.MEDIAN else x \
                            for (i, x) in enumerate(seg.start)]
            seg.end = [Segment.WEAVE if i >= seg.units.index(Segment.LANE) \
                        and i + 1 + len(CSURFactory.roadside[mode[0]]) < len(seg.start)
                        and x == Segment.MEDIAN else x \
                            for (i, x) in enumerate(seg.end)]
        return seg

   
class BaseAsset(Asset):
    def __init__(self, x0_start, *nlanes_start, median=1):
        super().__init__(x0_start, nlanes_start, medians=[median, median])

    def get_blocks(self):
        return self._blocks[0]

    def x0(self):
        return self.get_blocks()[0].x_left
    
    def x1(self):
        return self.get_blocks()[-1].x_right
    

class TwoWayAsset(Asset):
    def __init__(self, left, right, mirror=True, append_median=True):
        if mirror:
            self.left = reverse(left)
            self.left.roadtype = left.roadtype
        else:
            self.left = left
        self.right = right
        self._blocks = [self.left._blocks[1 - i] + self.right._blocks[i] for i in [0, 1]]
        self._infer_roadtype()
        self.append_median = append_median
    
    def _infer_roadtype(self):
        typestring = (self.left.roadtype + self.right.roadtype).strip("b")
        if len(typestring) > 1 and typestring[0] != typestring[1]:
            raise Exception("Invalid two-way construction!: %s,%s" % (self.left, self.right))
        self.roadtype = "b" if typestring == "" else typestring[0]
    
    def nl(self):
        return sum(x.nlanes for x in self._blocks[0])

    def get_all_blocks(self):
        # right hand traffic
        blocks_right = self.right._blocks
        blocks_left = [[x.mirror() for x in self.left._blocks[1]], [x.mirror() for x in self.left._blocks[0]]]
        return [blocks_left[0] + blocks_right[0], blocks_left[1] + blocks_right[1]]

    def is_twoway(self):
        return True

    def has_sidewalk(self):
        return Segment.SIDEWALK in self.right.get_model('g').start

    def has_bikelane(self):
        return Segment.BIKE in self.right.get_model('g').start

    def is_roundabout(self):
        return False

    def has_trafficlight(self):
        return True

    def n_central_median(self):
        if self.roadtype != 'b':
            raise NotImplementedError("central median count only avaiable for base module!")
        return [int(self.left.xleft[0] // SW.MEDIAN), int(self.right.xleft[0] // SW.MEDIAN)]
    
    def is_undivided(self):
        return self.get_model('g').undivided

    def center(self):
        return [(br[-1].x_right - bl[-1].x_right) / 2 for bl, br in zip(self.left._blocks, self.right._blocks)]

    def asym(self):
        return [self.right.nlanes[i][0] - self.left.nlanes[i][0] for i in [0, 1]]

    def get_model(self, mode='g'):
        if mode[-1] == 'c':
            seg = TwoWay(self.left.get_model(mode[0]), self.right.get_model(mode[0]), self.append_median)
            for u in [seg.left.start, seg.right.start, seg.left.end, seg.right.end]:
                i = 0
                while u[i] == Segment.MEDIAN:
                    u[i] = Segment.CHANNEL
        else:           
            seg = TwoWay(self.left.get_model(mode), self.right.get_model(mode), self.append_median)
        return seg

    def __str__(self):
        names = [twoway_reduced_name(x, y) for x, y in zip(self.left._blocks[::-1], self.right._blocks)]
        return combine_name(names)



            
