from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=False

def get_cores_per_node(args):
  return 40

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "srun "

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_mpirun_options_mpiomp(args):
  return f"-n {int(args.nodes * get_cores_per_node(args) / args.threads)}"

def get_submit_cmd(args):
  return "sbatch"

def get_env(args):
  env = """
module purge
#module load jemalloc/5.2.1/intel-19.0.3.199
module load gcc/9.2.0/gcc-4.8.5
#module load openmpi/4.0.2/gcc-9.2.0
module load openmpi/3.1.5/gcc-9.2.0
module load cmake/3.16.2/intel-19.0.3.199
module load python/3.7.6/intel-19.0.3.199
export PATH=$PATH:${HOME}/install/tbsla/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/install/hpx/2b91d44/release/lib64:${HOME}/install/boost/1.69.0/release/lib:${HOME}/install/hwloc/2.0.4/lib:${HOME}/install/tbsla/lib:${HOME}/install/tbsla/lib64
"""
  return env


def get_header(args):
  ncores = (get_cores_per_node(args) if args.lang != "HPX" else 1)  * args.nodes

  header = f"""\
#!/bin/bash
#SBATCH -p cpu_med
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={get_cores_per_node(args)}
#SBATCH --exclusive
#SBATCH --output logs/tbsla_%x.%J.out

export TBSLA_LOG_FILE=logs/tbsla_$SLURM_JOB_NAME.$SLURM_JOB_ID.out

"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ['TBSLA_LOG_FILE']
  return dic

def post_processing(args):
  s = ""
  return s

def post_run_cmd(args):
  s = ""
  return s

