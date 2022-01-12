import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hostfile", dest="hostfile", help="Hostfile to generate", type=str, required=True)
parser.add_argument("-n", dest="n", help="Number of nodes", type=int, required=True)
args = parser.parse_args()

nlines = 0

if os.path.isdir(args.hostfile):
  with open(args.hostfile, "r") as f:
    lines = f.readlines()
    nlines = len(lines)

if nlines != args.n:
  with open(args.hostfile, "w") as f:
    f.write("localhost\n" * args.n)
