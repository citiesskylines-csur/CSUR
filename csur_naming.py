EPS = 1e-6
LANEWIDTH = 3.75

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


'''
def offset_number(x):
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
            #centered = Carriageway(l.nlanes + r.nlanes, -l.x_right)
            if r.x_left == 0 and r.nlanes - l.nlanes == 1:
                suffix = 'S'
            else:
                #suffix = centered.suffix()
                suffix = 'C'
            reduced.append("%dD%s" % (l.nlanes + r.nlanes, suffix))
            i += 1
        elif str(l) == str(r):       
            reduced.append("%dD%s" % (2 * l.nlanes, l.suffix()))
            i += 1
    name_l = [str(x) for x in block_l[i:]]
    name_r = [str(x) for x in block_r[i:]]
    if not reduced:
        reduced = [DIRECTION_SEPERATOR]
    return name_l + reduced + name_r

def get_suffix(block, init_r):
    if block.get_offset() == 0:
        offset_code = 'C'
        return offset_code
    elif block.get_offset() == LANEWIDTH / 2 and block.nlanes > 1:
        offset_code = 'S'
        return offset_code
    #elif self.get_offset() == -Carriageway.init_r:
    #    offset_code = 'CL'
    #    return offset_code
    elif block.get_offset() > 0:
        offset_code = 'R'
    else:
        offset_code = 'L'
    if abs(block.x_left - init_r) < EPS \
            or abs(block.x_right + init_r) < EPS \
            or block.get_offset() == 0:
        n_offset = ''
    else:
        if block.x_right + block.x_left >= 0:
            n_offset = offset_number(block.x_right)
        else:
            n_offset = offset_number(LANEWIDTH - block.x_left)
    return offset_code + n_offset


