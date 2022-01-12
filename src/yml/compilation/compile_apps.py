import os
import subprocess
import sys
import argparse
import re

def execute_command(cmd):
  p = subprocess.Popen(cmd, shell=True)
  try:
    rv = p.wait(timeout = 200)
    p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()

def execute_command_pipe_stdout(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  try:
    rv = p.wait(timeout = 200)
    out = p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    out = p.communicate()
  return out

parser = argparse.ArgumentParser()
parser.add_argument("--C", dest="C", help="Cores per task", type=int, required=True)
parser.add_argument("--BGR", dest="BGR", help="Number of submatrices in the row dimension", type=int, required=True)
parser.add_argument("--BGC", dest="BGC", help="Number of submatrices in the column dimension", type=int, required=True)
parser.add_argument("--app", dest="app", help="Application", type=str, required=True, choices=['spmv', 'a_axpx'])
args = parser.parse_args()

# define the name of the directory to be created
path = "_yml_tmpdir/app"

if not os.path.isdir(path):
  try:
    os.makedirs(path)
  except OSError:
    print ("Creation of the directory %s failed" % path)
  else:
    print ("Successfully created the directory %s " % path)

filename = "src/yml/app/" + args.app + ".query"
app = args.app + f"_{args.C}_{args.BGR}_{args.BGC}"
apppath = path + "/" + app + ".query"
cmd = f"sed 's/NCORE/{args.C}/g;s/BGR/{args.BGR}/g;s/BGC/{args.BGC}/g' " + filename + " > " + apppath
execute_command(cmd)

os.chdir(path)
cmd = "yml_compiler " + app + ".query"
execute_command(cmd)

cmd = "omrpc-register-yml " + app + ".query.yapp"
out = execute_command_pipe_stdout(cmd)
out = out[0].decode('utf-8')
r = re.findall("^"+app+".*", out, re.M)
resc = []
for i in r:
  resc.append(i + "\n")

home = os.environ['HOME']
stub_file = home + "/.omrpc_registry/stubs"

if os.path.isfile(stub_file):
  with open(stub_file, "r") as fp:
    c = fp.readlines()
  c += resc
  c = list(dict.fromkeys(c))
  with open(stub_file, "w") as fp:
    fp.writelines(c)
else:
  with open(stub_file, "w") as fp:
    fp.writelines(resc)

