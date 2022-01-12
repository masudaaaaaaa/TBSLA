import subprocess
import os

def gen_submit_cmd(machine, args, sub_file_path, sub_file_content):
  if os.path.isfile(sub_file_path):
    os.remove(sub_file_path)

  with open(sub_file_path, 'w', encoding = 'utf-8') as f:
    f.write(sub_file_content)

  command = ""
  if machine.LOAD_ENV_BEFORE_SUBMIT:
    command += machine.get_env(args) + "\n"
  command += machine.get_submit_cmd(args) + " " + sub_file_path

  if os.path.isfile(sub_file_path):
    print(command)
    if args.dry == "False":
      p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
      print(p.communicate()[0].decode('utf-8'))
