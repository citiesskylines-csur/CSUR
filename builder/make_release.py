import os, sys
sys.path.append(os.getcwd())
from prefab import make

if sys.argv[5] == 'all':
    args = os.listdir('release/input/')
else:
    args = sys.argv[5:]

for arg in args:
    if '.' not in arg:
        pkg_key = arg
        arg += '.txt'
    else:
        pkg_key = arg.split('.')[0]
    with open(os.path.join('release', 'input', arg), 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        make(os.getcwd(), lines, output_path=os.path.join('release', 'import', pkg_key))

