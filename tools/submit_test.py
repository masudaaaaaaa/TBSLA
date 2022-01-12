import os
import sys
import argparse
import common.argparse as cap
import importlib
import subprocess

parser = argparse.ArgumentParser(description="Submit TBSLA test run to the job scheduler", parents=[cap.add_common(), cap.add_submit()])
parser.add_argument("--exe", dest="exe", help="Executable to run", type=str, required=True)
args = parser.parse_args()

machine = importlib.import_module("machine." + args.machine)

header = machine.get_header(args)
ncores = machine.get_cores_per_node(args) * args.nodes

dict_to_pass = vars(args)
dict_to_pass["cores"] = ncores
dict_to_pass["test"] = "test"

command = f'python tools/run.py --rod 0'
for k in ['resfile',  'machine', 'timeout']:
  command += f" --{k} {dict_to_pass[k]}"
command += ' " '

if args.lang == "MPI":
  command += machine.get_mpirun(args) + f" -n {ncores} {args.exe}"

if args.lang == "HPX":
  if args.nodes == 1:
    command += f"{args.exe}"
  else:
    command += machine.get_mpirun(args) + f" -n {args.nodes} {args.exe} -l {args.nodes}"

header += command +  f'" --dic "{dict_to_pass}"\n\n'

fname = f"submit_test_{args.lang}_n{args.nodes}.sh"

if os.path.isfile(fname):
  os.remove(fname)

with open(fname, 'w', encoding = 'utf-8') as f:
  f.write(header)

command = machine.get_env(args) + "\nllsubmit " + fname

if os.path.isfile(fname):
  print(command)
  print(args.dry)
  if args.dry == "False":
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    print(p.communicate()[0].decode('utf-8'))

