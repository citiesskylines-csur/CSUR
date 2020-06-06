import csur, builder
from builder import Builder
from assets import BaseAsset
from compiler import asset_from_name
builder.N_MEDIAN = 2
max_lane = 6

LW = csur.StandardWidth.LANE

codes_all = [['1', '2', '2P', '3', '3P', '4P', '5P', '6', '7'],
                ['2', '3', '4P', '5P', '6P', '7'],
                ['3', '4', '5P', '6P'],
                ['4', '5', '6P'],
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

codes_left =  [['-2P', '-2', '-1P', '-1', '-0P', '0'],
               ['-1P', '-1', '-0P', '0', '0P'],
               ['-0P', '0', '0P', '1'],
               ['-1P', '-1'],
               ['-2P']
              ]

#for i in range(len(codes_left)):
#    codes_insane[i] = codes_left[i] + codes_insane[i]



builder = Builder(codes_insane, max_undivided=6).build(twoway=True)
assetpack = builder.get_assets()
asset_list = []
for k in assetpack.keys():
    asset_list.extend(assetpack[k])

for key in assetpack.keys():
    for x in assetpack[key]:
        from_name = asset_from_name(str(x))
        assert str(from_name) == str(x), "Test failed! %s, %s" % (from_name, x)
print("%d tests passed." % len(asset_list))
        