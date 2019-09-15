import os, configparser
from graphics import *
from csur import typename, Segment
from csur import StandardWidth as SW
from assets import Asset, TwoWayAsset

EPS = 1e-6

icons = {
    'BASE': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
    'SHIFT': [(-0.35, -0.43), (0.5, -0.43), (0.35, 0.43), (-0.5, 0.43)],
    'TRANS': [(-0.4, -0.46), (0.4, 0), (-0.4, 0.46)],
    'RAMP':   [(0, -0.43), (0.5, 0.43), (-0.5, 0.43)]
    }

SIZE = 1000

LSPACE = 0.04
RSPACE = 0.04

ROOT = os.path.dirname(os.path.abspath(__file__))

def make_panel(canvas, roadtype, name, config, lspace=LSPACE, rspace=RSPACE):
    banner_margin = 0.005
    banner_height = 0.12
    banner_text_size = 0.08
    banner_icon_size = 0.05

    bg = config[roadtype]['background'].split(',')
    gradient = Gradient(0.0, 1.0, 1.0, 0.0)
    gradient.add_color(0, Color(bg[0])).add_color(1, Color(bg[1]))
    canvas.add_background(gradient)
    canvas.add_rectangle((banner_margin, banner_margin), (1-banner_margin, banner_height-banner_margin), Color(1.0))
    canvas.add_image(os.path.join(ROOT, "img/csur_logo.png"), (lspace, banner_height/2), height=banner_text_size, valign=Anchor.MIDDLE)
    if len(name) < 7:
        font = ('Roboto', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    else:
        font = ('Roboto Condensed', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    wtext, htext = canvas.add_text(name, (1 - rspace, banner_height/2), banner_text_size, Color(0.3), 
                                    valign=Anchor.MIDDLE, halign=Anchor.RIGHT, fontface=font)
    canvas.add_polygon(icons[roadtype], (1 - rspace - wtext - banner_icon_size*0.9, banner_height/2), 
                                            Color(config[roadtype]['icon']), scale=banner_icon_size)


def make_axis(canvas, asset, config, rspace=RSPACE, thumbmode=None, draw_reference=True):
    margin = 0.015
    axis_top = 0.22
    axis_bottom = 0.88
    axis_label_size = 0.075
    tick_height = 0.054

    seg = asset.get_model('g')
    blocks = asset.get_all_blocks()
    if asset.is_twoway():
        wmax = max(blocks[0][-1].x_right, -blocks[0][0].x_left,
                   blocks[1][-1].x_right, -blocks[1][0].x_left, 3.5*SW.LANE)
        units = [x or y for x, y in zip(seg.left.start[::-1] + seg.right.start, seg.left.end[::-1] + seg.right.end)]
        x_start = [-x for x in seg.left.x_start[::-1]] + seg.right.x_start[1:]
        x_end = [-x for x in seg.left.x_end[::-1]] + seg.right.x_end[1:]
        i0 = -int(wmax // (SW.LANE / 2))
        wmax *= 2
    else:
        wmax = max(blocks[0][-1].x_right, blocks[1][-1].x_right, 7.5*SW.LANE)     
        units = [x or y for x, y in zip(seg.start, seg.end)]
        x_start, x_end = seg.x_start, seg.x_end
        i0 = 0
    nticks = int(wmax // (SW.LANE / 2))
    axis_start = 1 - rspace - 15 * tick_height
    tick_length = tick_height * 15 / nticks
    colors_top = [230] * nticks
    colors_bottom = [230] * nticks
    axiscolor = config[typename[asset.roadtype]]['axis'].split(',')
    for i, b in enumerate(blocks[0]):
        p = int(b.x_left // (SW.LANE / 2)) - i0
        colors_bottom[p : p + b.nlanes*2] = [axiscolor[i % len(axiscolor)]] * b.nlanes*2
    for i, b in enumerate(blocks[1]):
        p = int(b.x_left // (SW.LANE / 2)) - i0
        colors_top[p : p + b.nlanes*2] = [axiscolor[i % len(axiscolor)]] * b.nlanes*2

    
    arrow_starts = []
    arrow_ends = []
    for i, u in enumerate(units):
        if u == Segment.LANE:
            x0 = axis_start + ((x_start[i + 1] + x_start[i]) / SW.LANE - i0) * tick_length
            x1 = axis_start + ((x_end[i + 1] + x_end[i]) / SW.LANE - i0) * tick_length
            arrow_starts.append(x0)
            arrow_ends.append(x1)
            if x_start[i + 1] == x_start[i]:
                x0 = x0 - tick_length if x0 < x1 else x0 + tick_length
            if x_end[i + 1] == x_end[i]:
                x1 = x1 - tick_length if x1 < x0 else x1 + tick_length
            r0 = (x0, axis_bottom - tick_height - margin)
            r1 = (x1, axis_top + tick_height + margin)
            if i >= seg.middle_index():
                canvas.add_line(r0, r1, 0.015, Color(1.0), arrow=1)
            else:
                canvas.add_line(r1, r0, 0.015, Color(1.0), arrow=1)  
    p = 0
    for i in range(i0, i0 + nticks):
        left = axis_start + (i - i0) * tick_length
        canvas.add_rectangle((left, axis_top), (left + tick_length, axis_top + tick_height), 
                            Color(colors_top[i - i0], a=0.5+0.5*(i % 2 != 0)))
        canvas.add_rectangle((left, axis_bottom - tick_height), (left + tick_length, axis_bottom),
                            Color(colors_bottom[i - i0], a=0.5+0.5*(i % 2 != 0)))
        if i % 2 == 0 and i > i0:
            canvas.add_text(str(abs(i) // 2), (left, axis_top - margin), axis_label_size, 
                    Color(colors_top[i - i0]), valign=Anchor.BOTTOM, halign=Anchor.CENTER)
            canvas.add_text(str(abs(i) // 2), (left, axis_bottom + margin), axis_label_size, 
                    Color(colors_bottom[i - i0]), valign=Anchor.TOP, halign=Anchor.CENTER)
            if draw_reference:
                alpha = 0
                if p < len(arrow_starts) and left >= min(arrow_starts[p], arrow_ends[p]):
                    p += 1
                if left - 2 * tick_length > max(arrow_starts[-1], arrow_ends[-1]) + EPS \
                    or left + 2*tick_length < min(arrow_starts[0], arrow_ends[0]) - EPS:
                    alpha = 0.5
                elif abs(left - 2*tick_length - max(arrow_starts[-1], arrow_ends[-1])) <= EPS \
                    or abs(left + 2*tick_length - min(arrow_starts[0], arrow_ends[0])) <= EPS:
                    alpha = 0.25
                elif 0 < p < len(arrow_starts):
                    alpha = 0
                else:
                    alpha = 0.125
                if alpha > 0:
                    r0 = (left, axis_bottom - tick_height - margin)
                    r1 = (left, axis_top + tick_height + margin)
                    # right hand traffic
                    if i > 0:
                        canvas.add_line(r0, r1, 0.015, Color(1.0,a=alpha), arrow=1)
                    elif i < 0:
                        canvas.add_line(r1, r0, 0.015, Color(1.0,a=alpha), arrow=1)
    
    if asset.is_undivided():
        c = config['UI']['strip']
        center = [axis_start + (x // (SW.LANE / 2) - i0) * tick_length for x in seg.center]
        canvas.add_line((center[0]-0.005, axis_bottom-tick_height-margin), 
                        (center[1]-0.005, axis_top+tick_height+margin), 
                        0.005, Color(c), arrow=0)
        canvas.add_line((center[0]+0.005, axis_bottom-tick_height-margin), 
                        (center[1]+0.005, axis_top+tick_height+margin), 
                        0.005, Color(c), arrow=0)
    
    if thumbmode in ['disabled', 'focused', 'pressed', 'hovered']:
        alpha = 0.9 if thumbmode == 'disabled' else 0.75
        canvas.add_background(Color(config['UI'][thumbmode], a=alpha))

    if thumbmode == 'hovered':
        accent = config[typename[asset.roadtype]]['icon']
        for i, u in enumerate(units):
            if u == Segment.LANE:
                x0 = axis_start + ((x_start[i + 1] + x_start[i]) / SW.LANE - i0) * tick_length
                x1 = axis_start + ((x_end[i + 1] + x_end[i]) / SW.LANE - i0) * tick_length
                if x_start[i + 1] == x_start[i]:
                    x0 = x0 - tick_length if x0 < x1 else x0 + tick_length
                if x_end[i + 1] == x_end[i]:
                    x1 = x1 - tick_length if x1 < x0 else x1 + tick_length
                canvas.add_line((x0, axis_bottom - tick_height - margin), 
                                (x1, axis_top + tick_height + margin), 
                                0.015, Color(accent), arrow=1)

        for i in range(i0, i0 + nticks):
            left = axis_start + (i - i0) * tick_length
            if colors_top[i - i0] != 230:
                canvas.add_rectangle((left, axis_top), (left + tick_length, axis_top + tick_height), 
                                    Color(accent, a=0.5+0.5*(i % 2 != 0)))
            if colors_bottom[i - i0] != 230:
                canvas.add_rectangle((left, axis_bottom - tick_height), (left + tick_length, axis_bottom),
                                    Color(accent, a=0.5+0.5*(i % 2 != 0)))
            if i % 2 == 0 and i != i0:
                if colors_top[i - i0] != 230:
                    canvas.add_text(str(abs(i) // 2), (left, axis_top - margin), axis_label_size, 
                            Color(accent), valign=Anchor.BOTTOM, halign=Anchor.CENTER)
                if colors_bottom[i - i0] != 230:
                    canvas.add_text(str(abs(i) // 2), (left, axis_bottom + margin), axis_label_size, 
                            Color(accent), valign=Anchor.TOP, halign=Anchor.CENTER)
        



def make_sidebar(canvas, asset, config, lspace=LSPACE/2):
    margin = 0.02
    bottom = 0.85
    width = 0.12
    bar_height = 0.027
    n_icons = 5
    icon_size = width - lspace
    canvas.add_rectangle((0, bottom), (width, bottom + bar_height), Color(1.0))
    x = lspace + icon_size / 2
    y_cur = bottom - (icon_size + margin) * n_icons + icon_size / 2
    alpha = 1 if asset.has_sidewalk() else 0.15
    canvas.add_image(os.path.join(ROOT, "img/sidewalk.png"), (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin
    alpha = 1 if asset.has_bikelane() else 0.15
    canvas.add_image(os.path.join(ROOT, "img/bike.png"), (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin
    alpha = 1 if asset.is_twoway() else 0.15
    canvas.add_image(os.path.join(ROOT, "img/twoway.png"), (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin*1.5
    alpha = 1 if asset.has_trafficlight() else 0.15
    canvas.add_image(os.path.join(ROOT, "img/trafficlight.png"), (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin

def draw(asset, configfile, filepath=None, mode=None):
    config = configparser.ConfigParser()
    config.read(configfile)
    roadtype = typename[asset.roadtype]

    canvas = Canvas(SIZE, SIZE)

    make_panel(canvas, roadtype, str(asset), config)
    make_sidebar(canvas, asset, config)
    if asset.is_twoway() and not asset.is_undivided() and str(asset.left) == str(asset.right):
        make_axis(canvas, asset.right, config, thumbmode=mode)
    else:
        make_axis(canvas, asset, config, thumbmode=mode)

    if filepath:
        suffix = '_' + mode + '.png' if mode else '_thumb.png'
        canvas.save(filepath + suffix)
    else:
        canvas.save(os.path.join(ROOT, "thumbnails/%s" % asset + suffix))

if __name__ == "__main__":
    '''
    from builder import Builder
    max_lane = 6
    codes_all = [['1', '2', '2P', '3', '3P', '4', '4P', '5P'],
                    ['2', '3', '4', '4P', '5P', '6P', '7'],
                    ['3', '4', '5P', '6P'],
                    ['4', '5', '6P'],
                    ['5', '6'],
                    ['6', '7'],
                ]
    builder = Builder(codes_all, MAX_UNDIVIDED=4).build()
    assetpack = builder.get_assets()
    asset_list = []
    for k in assetpack.keys():
        asset_list.extend(assetpack[k])
    print(len(asset_list))
    for asset in asset_list:
        if not asset.is_twoway():
            draw(asset)
    '''
    asset2 = Asset(3*1.875, 2)
    asset = Asset(-1.875, 4)
    asset = TwoWayAsset(asset2, asset)
    for mode in [None, 'disabled', 'hovered', 'focused', 'pressed']:
        draw(asset, "C:/Work/roads/CSUR/img/color.ini", "example", mode=mode)
    #'''