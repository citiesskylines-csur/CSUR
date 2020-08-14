import os, sys, argparse
sys.path.append(os.getcwd())

from prefab import make

if __name__ == "__main__": 
    custom_args = sys.argv[5:]
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('input', help='name of module', nargs='+')
    parser.add_argument('-r', '--reverse', action='store_true', help='also makes the reverse of an interface module')
    parser.add_argument('-o', '--output', default='output', 
        help='output file path')
    args = parser.parse_args(custom_args)
    assetlist = args.input
    if len(args.input) == 1 and os.path.isfile(args.input[0]):
        with open(args.input[0]) as f:
            assetlist = [x.strip() for x in f.readlines()]
    make(os.getcwd(), assetlist, reverse=args.reverse, output_path=args.output)