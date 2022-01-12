import re

def get_worker_number(filename):
  with open(filename, 'r+') as f:
    data = f.read()
    r = re.findall("pid=.*", data)
    return len(set(r))
  return 0

def get_task_number(filename):
  with open(filename, 'r+') as f:
    data = f.read()
    r = re.findall("SUCCESS", data)
    return len(r)
  return 0

def get_task_list(filename):
  with open(filename, 'r+') as f:
    data = f.read()
    r = re.findall("^----------------------------------------\sTask.*Finished[\w\s:()=.]*----------------------------------------$", data, re.M)
    return r
  return []

def get_component_name(task_text):
  r = re.search("^component[ ]*:.*\(", task_text, re.M)
  r = re.sub("^component[ ]*: ", "", r.group())
  r = re.sub(" \(", "", r)
  return r

def get_start_time(task_text):
  r = re.search("^start.*\n.*\)", task_text, re.M)
  r = re.search(".*\(.*\)", r.group(), re.M)
  r = re.sub(" *\(", "", r.group())
  r = re.sub("\)", "", r)
  return r

def get_end_time(task_text):
  r = re.search("^end.*\n.*\)", task_text, re.M)
  r = re.search(".*\(.*\)", r.group(), re.M)
  r = re.sub(" *\(", "", r.group())
  r = re.sub("\)", "", r)
  return r

def get_elapsed_time(task_text):
  r = re.search("^ela.*time:.*", task_text, re.M)
  r = re.sub("^ela.*time: ", "", r.group())
  return r

def is_task_ignored(taskname, tasks_to_ignore):
  for i in tasks_to_ignore:
    if taskname.startswith(i):
      return True
  return False

def insert_add_dict(dic, key, value):
    if key not in dic:
      dic[key] = value
    else:
      dic[key] += value

def insert_min_dict(dic, key, value):
    if key not in dic:
      dic[key] = value
    else:
      if dic[key] > value:
        dic[key] = value

def insert_max_dict(dic, key, value):
    if key not in dic:
      dic[key] = value
    else:
      if dic[key] < value:
        dic[key] = value

def get_task_(filename, tasks_to_ignore):
  tasks = get_task_list(filename)
  dic = dict()
  for t in tasks:
    r = get_component_name(t)
    if is_task_ignored(r, tasks_to_ignore): continue
    r = float(get_start_time(t))
    insert_min_dict(dic, "start_time", r)
    r = float(get_end_time(t))
    insert_max_dict(dic, "end_time", r)
    r = float(get_elapsed_time(t))
    insert_add_dict(dic, "elapsed_time", r)
  return dic
