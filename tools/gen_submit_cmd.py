import os
import sys
import argparse
import math
import importlib
import itertools
import math

parser = argparse.ArgumentParser()
parser.add_argument("--Ns", dest="Ns", help="Min Number of nodes", type=int, required=True)
parser.add_argument("--Ne", dest="Ne", help="Max Number of nodes", type=int, required=True)
parser.add_argument("--NR", dest="NR", help="Matrix size (rows)", type=int, required=True)
parser.add_argument("--NC", dest="NC", help="Matrix size (columns)", type=int, required=True)
parser.add_argument("--C", dest="C", help="C for CQMAT", type=int, required=True)
parser.add_argument("--NNZ", dest="NNZ", help="Non-zero ratio per row", type=float, default=0.0001, required=True)
parser.add_argument("--machine", dest="machine", help="Machine name", type=str, required=True)
parser.add_argument("--YML", dest="YML", help="Generate submission commands for YML", action='store_const', default=False, const=True)
parser.add_argument("--MPI", dest="MPI", help="Generate submission commands for MPI", action='store_const', default=False, const=True)
parser.add_argument("--MPIOMP", dest="MPIOMP", help="Generate submission commands for MPI+OpenMP", action='store_const', default=False, const=True)
parser.add_argument("--OMP", dest="OMP", help="Generate submission commands for OpenMP", action='store_const', default=False, const=True)
parser.add_argument("--HPX", dest="HPX", help="Generate submission commands for HPX", action='store_const', default=False, const=True)
parser.add_argument("--OP", dest="OP", help="Operation to execute", type=str, required=True)
parser.add_argument("--numa-init", dest="numainit", help="Call NUMAinit function that perform first touch memory allocation", action='store_const', default=False, const=True)
args = parser.parse_args()

C = args.C
OP = args.OP
OP_suffix = ""
if OP == "page_rank":
  OP_suffix = "_pagerank"
#MTYPE = 'cqmat'
MTYPE = "random_stoch"
#formats = {'COO', 'CSR', 'ELL', 'DENSE', 'SCOO'}
#formats = {'COO', 'CSR', 'ELL', 'SCOO'}
formats = {'CSR', 'ELL', 'SCOO'}
machine_informations = importlib.import_module("machine." + args.machine)
NODES = [int(math.pow(2, i)) for i in range(int(math.log2(args.Ns)), int(math.log2(args.Ne)) + 1)]
ncores = machine_informations.get_cores_per_node(None)

CPT = [ncores // 2, ncores, 2 * ncores, 3 * ncores]
BLOCKS = [1, 2, 4, 6, 8, 12, 16]
CPT_BLOCKS = list(itertools.product(CPT, BLOCKS, BLOCKS))
THREADS = [1, 2, 4, 6, 12, 24, 48]

timeout = 500
walltime = int(6 * timeout / 60) + 1

def decomp(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n >= 1:
    factors.append(n)
  return factors 

def decomp_pairs(n):
  pairs = []
  d = 1
  factors = decomp(n)
  for i in factors:
    d = d * i
    pairs.append((n // d, d))
    pairs.append((d, n // d))
  return sorted(set(pairs))

for n in NODES:
  print('# nb nodes : ', n)
  factors = decomp_pairs(n * ncores)
  for mf in formats:
    for f in factors:
      if args.MPI:
        print(f'python tools/submit{OP_suffix}.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --NNZ {args.NNZ} --machine {args.machine} --lang MPI --wall-time {walltime} --GR {f[0]} --GC {f[1]} --timeout {timeout} {"--numa-init" if args.numainit else ""}')
      if args.HPX:
        print(f'python tools/submit{OP_suffix}.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --NNZ {args.NNZ} --machine {args.machine} --lang HPX --wall-time {walltime} --GR {f[0]} --GC {f[1]} --timeout {timeout}')
      if args.YML:
        for cbb in CPT_BLOCKS:
          cpt = cbb[0]
          bgr = cbb[1]
          bgc = cbb[2]
          gr = f[0]
          gc = f[1]
          lgr = int(gr / bgr)
          lgc = int(gc / bgc)
          if lgc == 0 or lgr == 0 or lgc * lgr != cpt or gr != bgr * lgr or gc != bgc * lgc or cpt > n * ncores: continue
          print(f'python tools/submit{OP_suffix}.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --NNZ {args.NNZ} --machine {args.machine} --lang YML --wall-time {walltime} --GR {f[0]} --GC {f[1]} --CPT {cbb[0]} --BGR {cbb[1]} --BGC {cbb[2]} --LGC {lgc} --LGR {lgr} --compile --timeout {timeout}')
  if args.MPIOMP:
    for t in THREADS:
      factors = decomp_pairs(int(n * ncores / t))
      for mf in formats:
        for f in factors:
          print(f'python tools/submit{OP_suffix}.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes {n} --C {C} --NNZ {args.NNZ} --machine {args.machine} --lang MPIOMP --wall-time {walltime} --GR {f[0]} --GC {f[1]} --threads {t} --timeout {timeout} {"--numa-init" if args.numainit else ""}')
if args.OMP:
  for t in THREADS:
    for mf in formats:
      print(f'python tools/submit{OP_suffix}.py --NR {args.NR} --NC {args.NC} --op {OP} --format {mf} --matrixtype {MTYPE} --nodes 1 --C {C} --NNZ {args.NNZ} --machine {args.machine} --lang OMP --wall-time {walltime} --threads {t} --timeout {timeout} {"--numa-init" if args.numainit else ""}')
