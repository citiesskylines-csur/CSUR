import os, json
from builder import Builder, get_packages

codes = [['1', '2', '2P', '3', '3P', '4', '4P', '5P', '6', '7'],
                ['2', '3', '4P', '5P', '6P', '7'],
                ['3', '4', '4P', '5P', '6P'],
                ['4', '5', '5P', '6P'],
                ['5', '6'],
                ['6', '7'],
            ]

builder = Builder(codes, MAX_UNDIVIDED=4).build()
assetpack = builder.get_assets()
variants = builder.get_variants()

names = []
for k in assetpack.keys():
    names.extend([str(x) for x in assetpack[k]])
for key in variants.keys():
    for x in variants[key]:
        if key == 'brt':
            names.append(str(x))
        else:
            names.append(str(x) + '_' + key)

pkg = get_packages(assetpack, variants)

if not os.path.exists("release/input"):
    os.makedirs("release/input")

for pkg_key in pkg.keys():
    to_make = []
    for f in pkg[pkg_key]:
        for suffix in ['', '_compact', '_express', '_uturn']:
            if str(f) + suffix in names:
                to_make.append(str(f) + suffix + '\n')
    with open('release/input/%s.txt' % pkg_key, 'w') as f:
        f.writelines(to_make)
