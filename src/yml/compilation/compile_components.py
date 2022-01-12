import os
import subprocess
import sys
import argparse

def execute_command(cmd):
  p = subprocess.Popen(cmd, shell=True)
  try:
    rv = p.wait(timeout = 200)
    p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(1)
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()

def create_dir(path):
  if not os.path.isdir(path):
    try:
      os.makedirs(path)
    except OSError:
      print ("Creation of the directory %s failed" % path)
    else:
      print ("Successfully created the directory %s " % path)

parser = argparse.ArgumentParser()
parser.add_argument("--C", dest="C", help="Cores per task", type=int, required=True)
args = parser.parse_args()

# define the name of the directory to be created
path = "_yml_tmpdir/components/c" + str(args.C)

create_dir(path + "/abst")
create_dir(path + "/impl")

# r=root, d=directories, f = files
for r, d, f in os.walk("src/yml/abst"):
  for file in f:
    if file.endswith(".query"):
      filename = os.path.join(r, file)
      basename = os.path.basename(filename)
      cmd = "sed s/NCORE/" + str(args.C) + "/g " + filename + " > " + path + "/abst/" + basename
      execute_command(cmd)
      cmd = "yml_component --force " + path + "/abst/" + basename
      execute_command(cmd)

for r, d, f in os.walk("src/yml/impl"):
  for file in f:
    if file.endswith(".query"):
      filename = os.path.join(r, file)
      basename = os.path.basename(filename)
      cmd = 'sed "s/NCORE/' + str(args.C) + '/g;s/NDATA/' + str(args.C * 4) + '/g" ' + filename + " > " + path + "/impl/" + basename
      execute_command(cmd)
      cmd = "yml_component --force " + path + "/impl/" + basename
      execute_command(cmd)
