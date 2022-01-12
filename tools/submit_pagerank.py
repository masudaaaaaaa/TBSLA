import os
import sys
import argparse
import common.argparse as cap
import common.submission as sub
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA page rank perf run to the job scheduler", parents=[cap.init_pagerank(), cap.add_common(), cap.add_submit()])
args, rest = parser.parse_known_args()

parents = []
if args.lang == "MPIOMP":
  parents.append(cap.init_mpiomp())
if args.lang != "OMP":
  parents.append(cap.add_gcgr())
if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
  parents.append(cap.add_c())
if args.matrixtype == "random_stoch":
  parents.append(cap.add_random_stoch())
if len(parents) > 0:
  parser2 = argparse.ArgumentParser(parents=parents)
  parser2.parse_args(rest, args)

machine = importlib.import_module("machine." + args.machine)
header = machine.get_header(args)
ncores = machine.get_cores_per_node(args) * args.nodes

dict_to_pass = vars(args)
dict_to_pass["cores"] = ncores
#dict_to_pass["op"] = "pagerank"
dict_to_pass["op"] = "page_rank"

command = f'python tools/run.py'
for k in ['resfile',  'machine', 'timeout']:
  command += f" --{k} {dict_to_pass[k]}"
command += ' " '

if args.lang == "MPI":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_page_rank_mpi"

if args.lang == "MPIOMP":
  header += f"export OMP_NUM_THREADS={args.threads * args.tpc}\n"
  command += machine.get_mpirun(args) + " " + machine.get_mpirun_options_mpiomp(args) + " tbsla_perf_page_rank_mpi_omp"

if args.numainit == "True":
  command += " --numa-init"
command += f" --GR {args.GR}"
command += f" --GC {args.GC}"
command += f" --matrix {args.matrixtype}"
if args.matrixfolder != ".":
  command += f" --matrix_folder {args.matrixfolder}"
command += f" --format {args.format}"

if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
  command += f" --matrix-dim {args.NC}"
  command += f" --C {args.C}"

if args.matrixtype == "cqmat":
  incr = 1
  for s in range(1, 2):
    for q in QLIST:
      dict_to_pass["Q"] = q
      dict_to_pass["S"] = s
      header += command + f' --Q {q} --S {s}" --dic "{dict_to_pass}"'
      if hasattr(machine, 'OUT_DICT_FILE_CASES') and args.lang in machine.OUT_DICT_FILE_CASES and hasattr(machine, 'OUT_DICT_FILE') and machine.OUT_DICT_FILE != None:
        header += f' --infile ' + machine.OUT_DICT_FILE + f'.{incr}.0'
      header += '\n\n'
      incr += 1
else:
  header += command +  f'" --dic "{dict_to_pass}"'
  if hasattr(machine, 'OUT_DICT_FILE_CASES') and args.lang in machine.OUT_DICT_FILE_CASES and hasattr(machine, 'OUT_DICT_FILE') and machine.OUT_DICT_FILE != None:
    header += f' --infile ' + machine.OUT_DICT_FILE + '.1.0'
  header += '\n\n'

if args.matrixtype == "random_stoch":
  command += f" --matrix-dim {args.NC}"
  command += f" --NNZ {args.NNZ}"

print(command)

header += machine.post_processing(args) + "\n"
fname = f"submit_page_rank_{args.lang}_n{args.nodes}_{args.matrixtype}_{args.format}"
if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
  fname += f"_nr{args.NR}_nc{args.NC}_c{args.C}"
if args.lang != "OMP":
  fname += f"_gr{args.GR}_gc{args.GC}"
if args.lang == "MPI":
  fname += f".sh"
elif args.lang == "MPIOMP" or args.lang == "OMP":
  fname += f"_t{args.threads}.sh"
sub.gen_submit_cmd(machine, args, fname, header)
