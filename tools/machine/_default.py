LOAD_ENV_BEFORE_SUBMIT=False

def get_env(args):
  return ""


def get_header(args):
  return ""

def get_cores_per_node(args):
  return 1

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "mpirun"

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_mpirun_options_mpiomp(args):
  return f"-n {int(args.nodes * get_cores_per_node(args) / args.threads)}"

def get_additional_info(args):
  dic = dict()
  return dic

def post_processing(args):
  s = ""
  return s

def post_run_cmd(args):
  s = ""
  return s
