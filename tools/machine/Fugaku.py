from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False
OUT_DICT_FILE="${PJM_STDOUT_PATH}"
OUT_DICT_FILE_CASES=['MPI', 'MPIOMP']

def get_cores_per_node(args):
  return 48

def get_mpirun(args):
  return "mpirun"

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_mpirun_options_mpiomp(args):
  return f"-n {int(args.nodes * get_cores_per_node(args) / args.threads)}"

def get_submit_cmd(args):
  return "pjsub"

def get_env(args):
  env = """
module purge
module load lang/tcsds-1.2.31
export PATH=$PATH:${INSTALL_DIR}/tbsla/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INSTALL_DIR}/tbsla/lib
"""
  return env


def get_header(args):
  if args.lang == "HPX":
    ncores_per_nodes = 1
  elif args.lang == "MPIOMP":
    ncores_per_nodes = int(get_cores_per_node(args) / args.threads)
  else:
    ncores_per_nodes = get_cores_per_node(args)
  ncores = ncores_per_nodes * args.nodes
  if args.lang == "YML":
    ncores += 1

  header = f"""\
#!/bin/bash
#PJM -L  "node={args.nodes}"
#PJM -L  "rscgrp=small"
#PJM -L  "elapse={args.walltime}m"
#PJM --mpi "max-proc-per-node={ncores_per_nodes}"
#PJM -s
#PJM --appname TBSLA
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export FLIB_CPU_AFFINITY="12-59:1"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export PLE_MPI_STD_EMPTYFILE="off"

"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ.get('PJM_STDOUT_PATH', '')
  dic['OMPI_MCA_plm_ple_numanode_assign_policy'] = os.environ.get('OMPI_MCA_plm_ple_numanode_assign_policy', '')
  dic['FLIB_CPU_AFFINITY'] = os.environ.get('FLIB_CPU_AFFINITY', '')
  dic['GOMP_CPU_AFFINITY'] = os.environ.get('GOMP_CPU_AFFINITY', '')
  dic['OMP_PROC_BIND'] = os.environ.get('OMP_PROC_BIND', '')
  return dic

def post_processing(args):
  s = ""
  return s

def post_run_cmd(args):
  s = ""
  return s

