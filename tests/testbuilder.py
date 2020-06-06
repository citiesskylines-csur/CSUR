import csur, builder
from builder import Builder, get_packages
from assets import BaseAsset
from compiler import asset_from_name
builder.N_MEDIAN = 2
max_lane = 6


#non-uniform offset
'''
codes_all = [['1', '2', '2P', '3', '3P', '4', '4P', '5P'],
                ['2', '3', '4P', '5P', '6P', '7'],
                ['3', '4', '5P', '6P'],
                ['4', '5', '6P'],
                ['5', '6'],
                ['6', '7'],
            ]
'''

LW = csur.StandardWidth.LANE


codes_all = [['1', '2', '2P', '3', '3P', '4P', '5P', '6', '7'],
                ['2', '3', '4P', '5P', '6P', '7'],
                ['3', '4', '5P', '6P'],
                ['4', '5', '6P'],
                ['5', '6'],
                ['6', '7'],
            ]

codes_all2 = [['1', '2', '2P', '3', '3P', '4P', '5P', '6', '7'],
                ['2', '3', '4P', '5P', '6P', '7'],
                ['3', '4', '4P', '5P', '6P'],
                ['4', '5', '5P', '6P'],
                ['5', '6'],
                ['6', '7'],
            ]

codes_insane = [['1', '1P', '2', '2P', '3', '3P', '4', '4P', '5', '5P', '6', '6P', '7'],
                ['2', '2P', '3', '3P', '4', '4P', '5', '5P', '6', '6P', '7'],
                ['3', '3P', '4', '4P', '5', '5P', '6', '6P', '7'],
                ['4', '4P', '5', '5P', '6', '6P', '7'],
                ['5', '5P', '6', '6P', '7'],
                ['6', '6P', '7'],
            ]

codes_4L = [['1', '2', '2P', '3', '3P', '4', '4P'],
                ['2', '3', '4P'],
                ['3', '4'],
                ['4'],
            ]

codes_single = [ ['-2P', '-1P', '1', '2'],
                 ['-1P','0P','2'],
                 ['0P', '1P'],
                 ['1P'],
                ]


PRINT = False
PACKAGE = False

builder = Builder(codes_all2, MAX_UNDIVIDED=4).build(twoway=True)
assetpack = builder.get_assets()
asset_list = []
for k in assetpack.keys():
    asset_list.extend(assetpack[k])
variants = builder.get_variants()
print(len(asset_list))



if PRINT:
    lines = []
    for key in assetpack.keys():
        for x in assetpack[key]:
            line = str(x.get_model('e'))
            if x.is_twoway():
                line += ' ' + str(x.left) + ' ' + str(x.right)     
            print(line)
            lines.append(str(x) + '\n')
    print("variants")
    
    for key in variants.keys():
        print(key)
        for x in variants[key]:
            line = str(x.get_model('e'))
            if x.is_twoway():
                line += ' ' + str(x.left) + ' ' + str(x.right)     
            print(line)
            if key == 'brt':
                lines.append(str(x) + '\n')
            else:
                lines.append(str(x) + '_' + key + '\n')

if PACKAGE:
    import json
    pkg = get_packages(assetpack, variants)
    for k, v in pkg.items():
        print(k, 'size:', len(v))
        print(v)

#new_asset = BaseAsset(LW*6, 1)
new_asset = asset_from_name("4R3")
print(new_asset)
added = builder.get_dependency(new_asset)
print("%d dependents:" % len(added))
[print(x) for x in added]
new_asset = asset_from_name("5R4")
print(new_asset)
added = builder.get_dependency(new_asset)
print("%d dependents:" % len(added))
[print(x) for x in added]
