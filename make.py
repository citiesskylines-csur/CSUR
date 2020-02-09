import os, sys, argparse
sys.path.append(os.getcwd())
from assetmaker import make

if __name__ == "__main__": 
    custom_args = sys.argv[5:]
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('input', help='name of module', nargs='+')
    parser.add_argument('-r', '--reverse', action='store_true', help='also makes the reverse of an interface module')
    parser.add_argument('--interp', default=None, help='set interpolation type')
    args = parser.parse_args(custom_args)
    make(os.getcwd(), args.input, reverse=args.reverse, interpolation=args.interp)