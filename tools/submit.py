import os
import sys
import argparse
import common.argparse as cap
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA run to the job scheduler", parents=[cap.init_parser()])
args = parser.parse_args()

if args.op == "a_axpx" and args.NR != args.NC:
  print(f"The operation {args.op} needs a squarre matrix (with NR = NC).")
  sys.exit(1)

machine = importlib.import_module("machine." + args.machine)

header = machine.get_header(args)

if args.matrixtype == "cqmat":
  nbq = 10
  for s in range(1, 3):
    for q in range(0, nbq + 1):
      header += f"python tools/run.py --Q {q / nbq} --S {s}"
      for i in range(1, len(sys.argv)):
        header += " " + sys.argv[i]
      header += "\n"
else:
  header += "python tools/run.py"
  for i in range(1, len(sys.argv)):
    header += " " + sys.argv[i]
  header += "\n"

fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_nr{args.NR}_nc{args.NC}_{args.matrixtype}_c{args.C}"

if args.lang == "HPX":
  fname += f"__N{args.N}"

fname += ".sh"

if os.path.isfile(fname):
  os.remove(fname)

with open(fname, 'w', encoding = 'utf-8') as f:
  f.write(header)

command = machine.get_env(args) + "\nllsubmit " + fname

if os.path.isfile(fname):
  print(command)
  p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
  print(p.communicate()[0].decode('utf-8'))

