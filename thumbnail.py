import configparser
from graphics import *
from csur import typename, Segment
from csur import StandardWidth as SW
from assets import Asset

icons = {
    'BASE': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
    'SHIFT': [(-0.35, -0.43), (0.5, -0.43), (0.35, 0.43), (-0.5, 0.43)],
    'TRANS': [(-0.4, -0.46), (0.4, 0), (-0.4, 0.46)],
    'RAMP':   [(0, -0.43), (0.5, 0.43), (-0.5, 0.43)]
    }

SIZE = 1000

CONFIG_GLOBAL = configparser.ConfigParser()
CONFIG_GLOBAL.read('img/color.ini')

LSPACE = 0.04
RSPACE = 0.04

def make_panel(canvas, roadtype, name, config=CONFIG_GLOBAL, lspace=LSPACE, rspace=RSPACE):
    banner_margin = 0.005
    banner_height = 0.12
    banner_text_size = 0.08
    banner_icon_size = 0.05

    bg = config[roadtype]['background'].split(',')
    gradient = Gradient(0.0, 1.0, 1.0, 0.0)
    gradient.add_color(0, Color(bg[0])).add_color(1, Color(bg[1]))
    canvas.add_background(gradient)
    canvas.add_rectangle((banner_margin, banner_margin), (1-banner_margin, banner_height-banner_margin), Color(1.0))
    canvas.add_image("img/csur_logo.png", (lspace, banner_height/2), height=banner_text_size, valign=Anchor.MIDDLE)
    if len(name) < 7:
        font = ('Roboto', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    else:
        font = ('Roboto Condensed', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    wtext, htext = canvas.add_text(name, (1 - rspace, banner_height/2), banner_text_size, Color(0.3), 
                                    valign=Anchor.MIDDLE, halign=Anchor.RIGHT, fontface=font)
    canvas.add_polygon(icons[roadtype], (1 - rspace - wtext - banner_icon_size*0.9, banner_height/2), 
                                            Color(config[roadtype]['icon']), scale=banner_icon_size)


def make_axis(canvas, asset, config=CONFIG_GLOBAL, rspace=RSPACE):
    margin = 0.015
    axis_top = 0.2
    axis_bottom = 0.9
    axis_label_size = 0.054
    tick_size = 0.054
    nticks = 15
    axis_start = 1 - rspace - nticks * tick_size
    colors_top = [230] * nticks
    colors_bottom = [230] * nticks
    blocks = asset.get_all_blocks()
    axiscolor = config[typename[asset.roadtype]]['axis'].split(',')
    for i, b in enumerate(blocks[0]):
        p = int(b.x_left // (SW.LANE / 2))
        colors_bottom[p : p + b.nlanes*2] = [axiscolor[i % len(axiscolor)]] * b.nlanes*2
    for i, b in enumerate(blocks[1]):
        p = int(b.x_left // (SW.LANE / 2))
        colors_top[p : p + b.nlanes*2] = [axiscolor[i % len(axiscolor)]] * b.nlanes*2
    for i in range(nticks):
        left = axis_start + i * tick_size
        canvas.add_rectangle((left, axis_top), (left + tick_size, axis_top + tick_size), 
                            Color(colors_top[i], a=0.5+0.5*(i % 2 != 0)))
        canvas.add_rectangle((left, axis_bottom - tick_size), (left + tick_size, axis_bottom),
                            Color(colors_bottom[i], a=0.5+0.5*(i % 2 != 0)))
        if i > 0 and i % 2 == 0:
            canvas.add_text(str(i // 2), (left, axis_top - margin), axis_label_size, 
                    Color(colors_top[i]), valign=Anchor.BOTTOM, halign=Anchor.CENTER)
            canvas.add_text(str(i // 2), (left, axis_bottom + margin), axis_label_size, 
                    Color(colors_bottom[i]), valign=Anchor.TOP, halign=Anchor.CENTER)
    
    seg = asset.get_model('g')
    units = [x or y for x, y in zip(seg.start, seg.end)]
    for i, u in enumerate(units):
        if u == Segment.LANE:
            x0 = axis_start + (seg.x_start[i + 1] + seg.x_start[i]) / SW.LANE * tick_size
            x1 = axis_start + (seg.x_end[i + 1] + seg.x_end[i]) / SW.LANE * tick_size
            if seg.x_start[i + 1] == seg.x_start[i]:
                x0 = x0 - tick_size if x0 < x1 else x0 + tick_size
            if seg.x_end[i + 1] == seg.x_end[i]:
                x1 = x1 - tick_size if x1 < x0 else x1 + tick_size
            canvas.add_line((x0, axis_bottom - tick_size - margin), 
                            (x1, axis_top + tick_size + margin), 
                            0.015, Color(1.0), arrow=1)

def make_sidebar(canvas, asset, config=CONFIG_GLOBAL, lspace=LSPACE/2):
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
    canvas.add_image("img/sidewalk.png", (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin
    alpha = 1 if asset.has_bikelane() else 0.15
    canvas.add_image("img/bike.png", (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin
    alpha = 1 if asset.is_twoway() else 0.15
    canvas.add_image("img/twoway.png", (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin*1.5
    alpha = 1 if asset.has_trafficlight() else 0.15
    canvas.add_image("img/trafficlight.png", (x, y_cur), width=icon_size, valign=Anchor.MIDDLE, halign=Anchor.CENTER, alpha=alpha)
    y_cur += icon_size + margin





def draw(asset):
    roadtype = typename[asset.roadtype]

    canvas = Canvas(SIZE, SIZE)

    config = CONFIG_GLOBAL

    make_panel(canvas, roadtype, str(asset))
    make_axis(canvas, asset)
    make_sidebar(canvas, asset)

    canvas.save("thumbnails/%s.png" % asset) 

if __name__ == "__main__":
    from builder import Builder
    max_lane = 6
    codes_all = [['1', '2', '2P', '3', '3P', '4', '4P', '5P'],
                    ['2', '3', '4', '4P', '5P', '6P', '7'],
                    ['3', '4', '6P'],
                    ['5', '6P'],
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