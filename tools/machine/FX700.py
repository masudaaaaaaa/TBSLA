from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False

def get_cores_per_node(args):
  return 48

def get_mpirun(args):
  return "mpirun "

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_mpirun_options_mpiomp(args):
  return f"-n {int(args.nodes * get_cores_per_node(args) / args.threads)}"

def get_submit_cmd(args):
  return "sbatch"

def get_env(args):
  env = """
export PATH=$HOME/.local/bin:$HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
OPENMPI=/usr/mpi/gcc/openmpi-4.0.3rc4
export PATH=$PATH:${INSTALL_DIR}/tbsla/bin:$OPENMPI/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INSTALL_DIR}/tbsla/lib:$OPENMPI/lib
"""
  return env


def get_header(args):
  if args.lang == "HPX":
    ncores_per_nodes = 1
    cpus_per_task = 1
  elif args.lang == "MPIOMP":
    ncores_per_nodes = int(get_cores_per_node(args) / args.threads)
    cpus_per_task = args.threads
  else:
    ncores_per_nodes = get_cores_per_node(args)
    cpus_per_task = 1
  ncores = ncores_per_nodes * args.nodes
  if args.lang == "YML":
    ncores += 1

  header = f"""\
#!/bin/bash
#SBATCH --partition fx700-1
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={ncores_per_nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --output logs/%J.%x.out
#SBATCH --error logs/%J.%x.out
#SBATCH --job-name TBSLA
#SBATCH --time {args.walltime}

export TBSLA_LOG_FILE=logs/$SLURM_JOB_ID.$SLURM_JOB_NAME.out
export OMPI_MCA_plm_ple_numanode_assign_policy=share_band
#export FLIB_CPU_AFFINITY="12-59:1"
#export GOMP_CPU_AFFINITY="12-59:1"
export GOMP_CPU_AFFINITY="0-47:1"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ.get('TBSLA_LOG_FILE', '')
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

