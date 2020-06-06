import math
import cairo
from graphics import *

WIDTH, HEIGHT = 1000, 1000

BASE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
SHIFT = [(-0.35, -0.43), (0.5, -0.43), (0.35, 0.43), (-0.5, 0.43)]

canvas = Canvas(WIDTH, HEIGHT)

C0 = Color('416066')
C1 = Color('447857')

gradient = Gradient(0.0, 1.0, 1.0, 0.0)
gradient.add_color(0, C0).add_color(1, C1)

canvas.add_background(gradient)

MARGIN = 0.005
canvas.add_rectangle((MARGIN, MARGIN), (1.0-MARGIN*2, 0.12-MARGIN), Color(1.0))
canvas.add_image("img/csur_logo.png", (0.04, 0.06), height=0.08, valign=Anchor.MIDDLE)
canvas.add_text('12DR', (0.96, 0.06), 0.08, Color(0.3), valign=Anchor.MIDDLE, halign=Anchor.RIGHT)

canvas.add_line((0.2, 0.8), (0.2, 0.2), 0.02, Color(0.9), arrow=1)
canvas.add_line((0.2, 0.8), (0.3, 0.2), 0.02, Color(0.9), arrow=1)
canvas.add_polygon(BASE, (0.5, 0.06), Color('2CB51D'), scale=0.08)
canvas.add_image("img/sidewalk.png", (0.1, 0.2), height=0.08, valign=Anchor.MIDDLE, halign=Anchor.CENTER)

canvas.add_rectangle((0.1, 0.9), (0.15, 0.95), Color(1.0, a=0.5))
canvas.save("example.png")  # Output to PNG