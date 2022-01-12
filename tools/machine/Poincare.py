from datetime import datetime
import os

LOAD_ENV_BEFORE_SUBMIT=True

def get_cores_per_node(args):
  return 16

def get_sockets_per_node(args):
  return 1

def get_mpirun(args):
  return "mpirun -x PATH -x LD_LIBRARY_PATH"

def get_mpirun_options_hpx(args):
  return f"-n {args.nodes}"

def get_mpirun_options_mpiomp(args):
  return f"-n {int(args.nodes * get_cores_per_node(args) / args.threads)}"

def get_submit_cmd(args):
  return "llsubmit"

def get_env(args):
  env = """
module purge
module load python/anaconda3-2018.12
unset LD_PRELOAD

module load cmake/3.14.1-gnu54 gnu/7.3.0 openmpi/2.1.2_intel15.0.0_tm gdb/7.5.1 mkl/11.2
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0
export PATH=$PATH:${HOME}/install/tbsla/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/install/hpx/d3954a10947/release/lib64:${HOME}/install/boost/1.69.0/release/lib:${HOME}/install/jemalloc/5.2.0/lib:${HOME}/install/hwloc/2.0.4/lib:${HOME}/install/petsc/3.12.3/release/lib
"""
  return env


def get_header(args):
  ncores = (16 if args.lang != "HPX" else 1)  * args.nodes

  header = f"""\
#@ class            = clallmds+
#@ job_name         = logs/tbsla-{datetime.now().strftime("%Y%m%d_%H%M")}-{args.lang}
#@ total_tasks      = {ncores}
#@ node             = {args.nodes}
#@ wall_clock_limit = {args.walltime}
#@ output           = $(job_name).$(jobid).log
#@ error            = $(job_name).$(jobid).err
#@ job_type         = mpich
#@ environment      = COPY_ALL
#@ node_usage       = not_shared
#@ queue
#
"""

  header += get_env(args)
  header += "\n"

  return header

def get_additional_info(args):
  dic = dict()
  dic['log_file'] = os.environ['LOADL_STEP_OUT']
  return dic

def post_processing(args):
  s = ""
  return s

def post_run_cmd(args):
  s = ""
  return s
