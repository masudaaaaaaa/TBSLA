from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False

def get_cores_per_node(args):
  return 24

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "mpirun"

def get_mpirun_options_hpx(args):
  return "-ppn 1"

def get_mpirun_options_mpiomp(args):
  return f"-ppn {int(get_cores_per_node(args)/args.threads)} -genv OMP_NUM_THREADS={args.threads * args.tpc}"

def get_submit_cmd(args):
  return "bsub <"

def get_env(args):
  env = """
module purge
module load gcc/8.3.0
module load intel-mpi/2020U1
module load anaconda3/5.1.0
export PATH=$PATH:${INSTALL_DIR}/tbsla/bin:${INSTALL_DIR}/yml/230/release/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INSTALL_DIR}/tbsla/lib:${INSTALL_DIR}/tbsla/lib64:${INSTALL_DIR}/hpx/28ceb04/release/lib64:${INSTALL_DIR}/boost/1.69.0/release/lib:${INSTALL_DIR}/jemalloc/5.2.0/lib:${INSTALL_DIR}/hwloc/2.0.4/lib:${INSTALL_DIR}/yml/230/release/lib
"""
  if args.lang == "YML":
    env += "export FI_MLX_ENABLE_SPAWN=yes\n"
    env += "export HOME=/workrd/SCR/NUM/jerome\n"
  return env


def get_header(args):
  if args.lang == "HPX":
    ncores_per_nodes = get_sockets_per_node(args)
  elif args.lang == "MPIOMP":
    ncores_per_nodes = int(get_cores_per_node(args)/args.threads)
  else:
    ncores_per_nodes = get_cores_per_node(args)
  ncores = ncores_per_nodes * args.nodes
  if args.lang == "YML":
    ncores += 1

  header = f"""\
#!/bin/bash
#BSUB -q P2_RD
#BSUB -R "span[ptile={ncores_per_nodes}]"
#BSUB -n {ncores}
#BSUB -o logs/tbsla_%J.out
#BSUB -e logs/tbsla_%J.err
#BSUB -J TBSLA_{args.lang}
#BSUB -x
#BSUB -W {args.walltime}

"""
  if args.lang == "YML":
    header += "#BSUB -w ended(TBSLA_YML)\n"
  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ.get('LSB_OUTPUTFILE', '')
  return dic

def post_processing(args):
  s = "rm core.*\n"
  if args.lang == "YML":
    s += "bkill ${LSB_BATCH_JID}\n"
  s += "exit\n"
  return s

def post_run_cmd(args):
  s = ""
  return s
