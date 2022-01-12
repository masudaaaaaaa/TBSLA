import argparse
import common.argparse as cap
import common.parse_yml_log as pyl
import common.exec_cmd as ec
import subprocess
import importlib
import re
import json
import sys
from datetime import datetime
import time
import shutil
import os

def find_match(match, string):
  r = re.search(match + ".*$", string, re.M)
  if r == None:
    return None
  f = r.group()
  if f.startswith(match):
    return f[len(match):]
  return None

parser = argparse.ArgumentParser(description="Run TBSLA application", parents=[cap.add_common(required = True)])
parser.add_argument('cmd')
parser.add_argument('--dic', dest="dic")
parser.add_argument('--rod', '--require-output-dic', help="Require an output dict from the app to check if it worked", dest="rod", type=int, default=1, choices=[0, 1])
args = parser.parse_args()

machine = importlib.import_module("machine." + args.machine)

print()
print()
print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), " ::: ", args.cmd)
print(args.dic)

start = time.time()
success = "true"
reason = ""

p = subprocess.Popen(args.cmd, bufsize = 0, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

t_begin = time.time()
seconds_passed = 0
match_output_pack = ''
match_working_dir = ''
while p.poll() is None and seconds_passed < args.timeout:
  time.sleep(0.1)
  seconds_passed = time.time() - t_begin
  stdout = ec.nonBlockRead(p.stdout)
  stderr = ec.nonBlockRead(p.stderr)
  print(stdout, end = '')
  print(stderr, end = '')
  if match_output_pack == '':
    match_found = find_match("Output pack: ", stdout)
    if match_found != None:
      match_output_pack = match_found
  if match_working_dir == '':
    match_found = find_match("Working Dir         : ", stdout)
    if match_found != None:
      match_working_dir = match_found

  if match_output_pack != '' and os.path.isfile(match_output_pack):
    break

if seconds_passed >= args.timeout:
  try:
    p.stdout.close()
    p.stderr.close()
    p.kill()
  except:
    pass
end = time.time()

if match_output_pack == '':
  success = "false"
  if reason != "":
    reason += " + "
  reason += "Output pack path not found during parsing"
if match_working_dir == '':
  success = "false"
  if reason != "":
    reason += " + "
  reason += "Working Dir path not found during parsing"

dic = dict()
for k, v in json.loads(str(args.dic).replace("'", '"')).items():
  if k in dic:
    dic[k + "_out"] = v
  else:
    dic[k] = v

dic["output_pack"] = match_output_pack
dic["working_dir"] = match_working_dir

if not os.path.isfile(dic["output_pack"]):
  success = "false"
  if reason != "":
    reason += " + "
  reason += "Output pack not found"
else:
  os.remove(dic["output_pack"])

log_file = dic["working_dir"] + "/exec_log"
if os.path.isfile(log_file):
  dic["worker_number"] = pyl.get_worker_number(log_file)
  dic["task_number"] = pyl.get_task_number(log_file)
  if dic["task_number"] < 1:
    success = "false"
    if reason != "":
      reason += " + "
    reason += "No tasks executed"
  n = pyl.get_task_(log_file, ['gen_'])
  dic["time_op"] = n.get("end_time", 0) - n.get("start_time", 0)
  dic["elapsed_time_op"] = n.get("elapsed_time", 0)
  n = pyl.get_task_(log_file, [])
  dic["time_app_in"] = n.get("end_time", 0) - n.get("start_time", 0)
  dic["elapsed_time_app_in"] = n.get("elapsed_time", 0)
else:
  success = "false"
  if reason != "":
    reason += " + "
  reason += "Log file not found"

dic["success"] = success
dic["false_reason"] = reason
dic["date"] = datetime.now().strftime("%Y%m%d_%H%M%S")
dic["time_app_out"] = (end - start)

if os.path.isdir(dic["working_dir"]):
  shutil.rmtree(dic["working_dir"])

dic.update(machine.get_additional_info(args))
dic.update(vars(args))

print(json.dumps(dic))
print(json.dumps(dic), file=open(args.resfile, 'a'))
