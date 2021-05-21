import os
from core.assets import Asset, BaseAsset, TwoWayAsset
from core.csur import LANEWIDTH, DIRECTION_SEPERATOR, SEGMENT_END_SEPERATOR, \
    offset_x

def asset_from_name(name, reverse=False):
    print("Compiling", name)
    is_twoway = 'D' in name or DIRECTION_SEPERATOR in name
    if SEGMENT_END_SEPERATOR in name:
        name = name.split(SEGMENT_END_SEPERATOR)
        if reverse:
            name = name[::-1]
        is_base = False
    else:
        is_base = True
    if is_twoway:
        if is_base:
            block_l, block_r = decode_twoway(name)
            return TwoWayAsset(asset_from_blocks(block_l), asset_from_blocks(block_r))
        else:
            block_l0, block_r0 = decode_twoway(name[0])
            block_l1, block_r1 = decode_twoway(name[1])
            return TwoWayAsset(asset_from_blocks(block_l0, block_l1), asset_from_blocks(block_r0, block_r1))
    else:
        if is_base:
            return asset_from_blocks(decode(name)) 
        else:
            return asset_from_blocks(decode(name[0]), decode(name[1]))

def asset_from_blocks(block_start, block_end=None):
    if block_end:
        x0, n0, m0 = parse_blocks(block_start)
        x1, n1, m1 = parse_blocks(block_end)
        return Asset(x0, n0, x1, n1, medians=[m0, m1])
    else:
        x, n, m = parse_blocks(block_start)
        return BaseAsset(x, *n, median=m)

def decode_twoway(name):
    if DIRECTION_SEPERATOR in name:
        name = name.split(DIRECTION_SEPERATOR)
        return tuple([decode(name[0]), decode(name[1])])
    else:
        block_all = decode(name)
        block_masked = decode(name, twoway_mask=True)
        block_l = []
        block_r = []
        extra_l = []
        extra_r = []
        on_right = False
        for b, bm in zip(block_all, block_masked):
            #print(b)
            if bm is not None:
               extra_r.append(b) if on_right else extra_l.append(b)
            else:
                on_right = True
                xleft, nl = b
                if xleft < 0:
                    split_loc = xleft + (nl // 2) * LANEWIDTH
                    block_l.append((-split_loc, nl // 2))
                    block_r.append((split_loc, (nl + 1) // 2))
                else: 
                # simplest case: mDRn
                    if nl % 2 != 0:
                        raise ValueError("Invalid two-way naming: %s!" % name)
                    nl //= 2
                    block_l.append((xleft, nl))
                    block_r.append((xleft, nl))
        #print(block_l + extra_l, block_r + extra_r)
        return block_l + extra_l, block_r + extra_r         
    
def decode(name, twoway_mask=False):
    blocks = []
    xleft_cur = None
    nl_cur = None
    # the string is parsed using a two-pointer algorithm
    p1 = p2 = 0
    twoway = False
    while (p1 <= len(name) and p2 <= len(name)):
        if nl_cur is None:
            twoway = False
            if p1 >= len(name):
                break
            while p2 < len(name) and '0' <= name[p2] <= '9':
                p2 += 1
            nl_cur = int(name[p1:p2])
        elif xleft_cur is None:
            if p1 >= len(name):
                break
            location = name[p1]
            p1 += 1
            if name[p1 - 1] == 'D':
                location = name[p1]
                p1 += 1
                twoway = True
            p2 = p1
            # only support 1-digit position!
            if p2 < len(name) and '1' <= name[p2] <= '9':
                p2 += 1
                if p2 < len(name) and name[p2] == 'P':
                    p2 += 1
            elif p2 < len(name) and name[p2:p2 + 2] == '0P':
                p2 += 2
            if p2 < len(name) and (name[p2] < '1' or name[p2] > '9'):
                p2 -= 1
            nl_single = nl_cur // 2 if twoway else nl_cur
            offset_code = str(nl_single) if p1 == p2 else name[p1:p2]
            #print(p1, p2, location, offset_code)
            if location == 'R':
                xleft_cur = offset_x(offset_code) - nl_single * LANEWIDTH
            elif location == 'L':
                xleft_cur = -offset_x(offset_code)
            elif location == 'C':
                xleft_cur = -nl_cur / 2 * LANEWIDTH
            elif location == 'S':
                xleft_cur = (1 - nl_cur) / 2 * LANEWIDTH
            elif location == 'U':
                xleft_cur = -(nl_cur / 2 + 0.25) * LANEWIDTH
            else:
                raise NotImplementedError("Offset location not understood! The offset suffix must be either of four letters in 'CSUR'.")
        else:
            #print('add')
            blocks.append(None if twoway and twoway_mask else (xleft_cur, nl_cur))
            xleft_cur = None
            nl_cur = None
        p1 = p2
    #print(blocks)
    return blocks

def parse_blocks(blocks):
    #print(blocks)
    xleft = blocks[0][0]
    nl = [blocks[0][1]]
    median = None
    for i in range(1, len(blocks)):
        nl.append(blocks[i][1])
        sep = blocks[i][0] - blocks[i - 1][0] - blocks[i - 1][1] * LANEWIDTH
        if sep < 0:
            print(blocks)
            raise ValueError("Overlapping lane blocks!")
        median_cur = int(sep // (LANEWIDTH / 2))
        if median is None:
            median = median_cur
        elif median_cur != median:
            raise ValueError("Inconsistent number of medians between blocks!")
    median = median or 1
    return xleft, nl, median


#blocks = decode_twoway("1R47DS")
#print(asset_from_name("5C"))