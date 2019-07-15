import matplotlib.pyplot as plt
from csur import Segment

DPI = 150

LENGTH = 64

# colors of each building unit
colors = ["1", 
            "0.3", 
            [0.5, 0.55, 0.48], 
            "0.5", 
            [0.3, 0.77, 0.25], 
            "0.7",
            "0.7",
            "0.6"]

def plot_polygon(ax, xs, dx, **kwargs):
        points = [[xs[0], 0], [xs[1], LENGTH], [xs[1] + dx[1], LENGTH], [xs[0] + dx[0], 0]]
        return ax.add_patch(plt.Polygon(points, **kwargs))
    
def plot_dashed_line(ax, xs, line_part=[0, 1], **kwargs):
    return ax.plot([xs[0] + (xs[1] - xs[0]) * line_part[0], xs[0] + (xs[1] - xs[0]) * line_part[1]],
                    [LENGTH * line_part[0], LENGTH * line_part[1]],
                    color="1", ls='--', dashes=(10, 8))

def draw(segment, ax=None):
    if not ax:
        plt.figure(dpi=DPI)
        ax = plt.gca()
    ax.set_aspect('equal', 'box')
    # Draw polygons
    for x0, c0, x1, c1 in zip(segment.x_start[:-1], segment.start, segment.x_end[:-1], segment.end):
        plot_polygon(ax, [x0, x1], [Segment.widths[c0], Segment.widths[c1]],
                                color=colors[c0 or c1])
    for i in range(1, len(segment.start)):
        if (segment.start[i - 1] or segment.end[i - 1]) == 1 and (segment.start[i] or segment.end[i]) == 1:
            line_part = [0, 1]
            if not (segment.start[i - 1] and segment.start[i]):
                line_part[0] = 0.5
            if not (segment.end[i - 1] and segment.end[i]):
                line_part[1] = 0.5    
            plot_dashed_line(ax, [segment.x_start[i], segment.x_end[i]], line_part)
    if not ax:
        plt.show()

if __name__ == "__main__":
    import csur, builder
    from builder import generate_all
    builder.N_MEDIAN = 1
    max_lane = 6
    #non-uniform offset
    '''
    codes_all = [['3', '4', '4P', '5', '5P', '6P', '7P', '8', '9'],
                 ['4', '5', '5P', '6P', '7P', '9'],
                 ['5', '6P', '7P', '9'],
                 ['6', '7P', '9'],
                 ['7', '9'],
                 ['8', '9'],
                ]
    '''

    

    codes_all = [['3', '4', '4P', '5', '5P', '6', '6P', '7P'],
                 ['4', '5', '6', '6P', '7P', '8P', '9'],
                 ['5', '6', '7P', '8P'],
                 ['6', '7', '8P'],
                 ['8'],
                 ['8', '9'],
                ]

    codes_u = ['5', '5P', '6', '6P', '7', '7P', '8', '8P','9']

    assetpack = generate_all(max_lane, codes_all=codes_all)
    asset_list = []
    for k in assetpack.keys():
        asset_list.extend(assetpack[k])
    #for o in asset_list:
    #    print(o.obj)
    print(len(asset_list))
    for key in assetpack.keys():
        print('Type: %s' % key)
        line = [[] for _ in range(max_lane)]
        for x in assetpack[key]:
            line[x.nl() - 1].append(str(x.get_model()))
        for i, l in enumerate(line):
            if l != []:
                print('number of lanes: %d' % (i + 1))
                print('\t'.join(l), '\n')
        print('-'*30)

    
    n_medians = [0, 0]
    for s in assetpack['ramp']: 
        x = 1 if len(s.get_blocks()[1]) == 2 else 0
        i = int((s.get_blocks()[x][1].x_left - s.get_blocks()[x][0].x_right) / 1.875)
        n_medians[i-1] += 1
    print(n_medians)
    '''
    #seg = csur.CSURFactory(mode='e', roadtype='a').get(3.75/2,6,3,1)
    seg = csur.CSURFactory(mode='e', roadtype='r').get([3.75*0.5, 3.75*0.5], [3,[3,1]])
    print(seg)
    '''
    #plt.figure(dpi=150)
    #draw(assetpack['access'][1].obj, plt.gca())
    #plt.show()