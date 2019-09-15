import math
import cairo

# arrows should point right
ARROW_PATH = [(0, 0), (3.68, -0.81), (7.21, -2.21), (7.28, -2.16), (5.97, 0), 
          (7.28, 2.16), (7.21, 2.21), (3.68, 0.81)]

def get_dim(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return max(xs) - min(xs), max(ys) - min(ys)

class Anchor:
    TOP = 0
    MIDDLE = 1
    BOTTOM = 2
    LEFT = 3
    CENTER = 4
    RIGHT = 5

    def get_topleft(position, size, anchors):
        anchor_v, anchor_h = anchors
        x, y = position
        if anchor_v == Anchor.TOP:
            top = y
        elif anchor_v == Anchor.MIDDLE:
            top = y - size[1] / 2
        elif anchor_v == Anchor.BOTTOM:
            top = y - size[1]
        else:
            raise ValueError("Invalid vertical anchor")
        if anchor_h == Anchor.LEFT:
            left = x
        elif anchor_h == Anchor.CENTER:
            left = x - size[0] / 2
        elif anchor_h == Anchor.RIGHT:
            left = x - size[0]
        else:
            raise ValueError("Invalid horizontal anchor")
        return left, top

class Color:

    ishex = lambda s: type(s) == str and len(s) == 6 and all('0' <= c <= '9' or 'A' <= c <= 'F' for c in s.upper())
    hex2rgb = lambda h: tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    def __init__(self, *args, a=1):
        self.r = self.g = self.b = None
        if len(args) == 1:
            if Color.ishex(args[0]):
                self.r, self.g, self.b = Color.hex2rgb(args[0])
            elif type(args[0]) == int:
                self.r = self.g = self.b = args[0] / 255
            elif type(args[0]) == float:
                self.r = self.g = self.b = args[0]
        elif len(args) == 3:
            if type(args[0]) == int:
                self.r = args[0] / 255
                self.g = args[1] / 255
                self.b = args[2] / 255
            if type(args[0]) == float:
                self.r, self.g, self.b = args
        if any(x is None or x < 0 or x > 1 for x in [self.r, self.g, self.b]):
            raise ValueError("Color type not understood, use HEX, monochrome or RGB")
        self.a = a

    def __str__(self):
        return str((self.r, self.g, self.b, self.a))
    
    def __repr__(self):
        return '<Color: %s>' % str(self)
    
    def __eq__(self, other): 
        return self.r == other.r and self.g == other.g and self.b == other.b and self.a == other.a 
    
    def pattern(self):
        return cairo.SolidPattern(self.r, self.g, self.b, self.a)

class Gradient:

    def __init__(self, *args, gradienttype='linear'):
        if gradienttype == 'linear':
            self._pattern = cairo.LinearGradient(*args)
        elif gradienttype == 'radial':
            self._pattern = cairo.RadialGradient(*args)
        else:
            raise NotImplementedError
    
    def add_color(self, position, color):
        self._pattern.add_color_stop_rgba(position, color.r, color.g, color.b, color.a)
        return self

    def pattern(self):
        return self._pattern


class Canvas:

    DEFAULT_FONT = ('Helvetica Neue', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)

    def __init__(self, xdim, ydim, canvastype='image'):
        self.objects = []
        self.xdim = xdim
        self.ydim = ydim
        if canvastype == 'image':
            self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, xdim, ydim)
        else:
            raise NotImplementedError
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(min(xdim, ydim), min(xdim, ydim))

    def save(self, path, fileformat='png'):
        if fileformat == 'png':
            self.surface.write_to_png(path)


    def add_background(self, color):
        self.add_rectangle((0, 0), (1, 1), color)

    def add_rectangle(self, topleft, bottomright, color):
        x0, y0 = topleft
        x1, y1 = bottomright
        self.ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
        if type(color) == Color or type(color) == Gradient:
            self.ctx.set_source(color.pattern())
        else:
            self.ctx.set_source(color)
        self.ctx.fill()
        self.ctx.move_to(0, 0)

    def add_polygon(self, polygon, position, color, scale=1, rotation=0):
        x, y = position
        self.ctx.save()
        self.ctx.translate(x, y)
        self.ctx.rotate(rotation)
        self.ctx.scale(scale, scale)
        if type(color) == Color or type(color) == Gradient:
            self.ctx.set_source(color.pattern())
        else:
            self.ctx.set_source(color)
        self.ctx.move_to(polygon[0][0],  polygon[0][1])
        for x, y in polygon[1:]:
            self.ctx.line_to(x, y)
        self.ctx.close_path()
        self.ctx.fill()
        self.ctx.restore()
        self.ctx.move_to(0, 0)

    def add_line(self, start, end, width, color, arrow=0):
        x0, y0 = start
        x1, y1 = end
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        width /= length
        if x1 == x0:
            angle = math.pi / 2 if y1 > y0 else -math.pi / 2
        else:
            angle = math.atan((y1 - y0) / (x1 - x0))
            if x1 < x0:
                angle += math.pi
        frac = 0
        if arrow:
            frac = arrow * get_dim(ARROW_PATH)[0] * width / 2
            while (frac > 1):
                 print("Warning: Arrow too large, decrease size by half")
                 arrow /= 2
                 frac = arrow * get_dim(ARROW_PATH)[0] * width / 2
        self.ctx.save()
        self.ctx.translate(x0, y0)
        self.ctx.rotate(angle)
        self.ctx.scale(length, length)
        self.ctx.move_to(0, -width/2)
        self.ctx.line_to(0, width/2)
        self.ctx.line_to(1 - frac, width/2)
        self.ctx.line_to(1 - frac, -width/2)
        self.ctx.close_path()
        if type(color) == Color or type(color) == Gradient:
            self.ctx.set_source(color.pattern())
        else:
            self.ctx.set_source(color)
        if arrow:
            self.ctx.move_to(1 + ARROW_PATH[0][0] * width * arrow, ARROW_PATH[0][1] * width * arrow)
            for x, y in ARROW_PATH[1:]:
                self.ctx.line_to(1 - x * width * arrow, y * width * arrow)
            self.ctx.close_path()
        self.ctx.fill()
        self.ctx.restore()
        self.ctx.move_to(0, 0)

    def add_image(self, path, position, width=None, height=None, valign=Anchor.TOP, halign=Anchor.LEFT, alpha=1, fileformat='png'):
        if fileformat == 'png':
            img = cairo.ImageSurface.create_from_png(path)
        else:
            raise NotImplementedError
        img_height = img.get_height()
        img_width = img.get_width()
        width_ratio = height_ratio = 1e8
        if width:
            width_ratio = width / float(img_width)
        if height:
            height_ratio = height / float(img_height)
        if width or height:
            scale_xy = min(height_ratio, width_ratio)
            img_height *= scale_xy
            img_width *= scale_xy
        else:
            raise ValueError('Width or height should be specified!')
        # scale image and add it
        self.ctx.save()
        left, top = Anchor.get_topleft(position, (img_width, img_height), (valign, halign))
        self.ctx.translate(left, top)
        self.ctx.scale(scale_xy, scale_xy)
        self.ctx.set_source_surface(img)
        self.ctx.paint_with_alpha(alpha)
        self.ctx.restore()


    def add_text(self, text, position, size, color, fontface=DEFAULT_FONT, valign=Anchor.BOTTOM, halign=Anchor.LEFT):
        typeface, slant, weight = fontface
        self.ctx.set_font_size(size)
        self.ctx.select_font_face(typeface, slant, weight)
        self.ctx.set_source_rgb(color.r, color.g, color.b)
        x, y, width, height, dx, dy = self.ctx.text_extents(text)
        # text is anchored at bottom by default
        left, top = Anchor.get_topleft((position[0], position[1] + height), (width, height), (valign, halign))
        self.ctx.move_to(left, top)
        self.ctx.show_text(text)
        self.ctx.stroke()
        return width, height


    


    
