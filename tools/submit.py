import os
import sys
import argparse
import common.argparse as cap
import common.exec_cmd as exe
import common.submission as sub
import importlib

NBQ = 6
QLIST = [q /(NBQ - 1) for q in range(0, NBQ)]

parser = argparse.ArgumentParser(description="Submit TBSLA perf run to the job scheduler", parents=[cap.init_parser(), cap.add_common(), cap.add_submit()])
args, rest = parser.parse_known_args()

parents = []
if args.lang == "MPIOMP":
  parents.append(cap.init_mpiomp())
if args.lang == "OMP":
  parents.append(cap.init_omp())
  if args.nodes != 1:
    print("OMP supports only one node")
    sys.exit(1)
if args.lang != "OMP":
  parents.append(cap.add_gcgr())
if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
  parents.append(cap.add_c())
if args.lang == "YML":
  parents.append(cap.init_yml())
if len(parents) > 0:
  parser2 = argparse.ArgumentParser(parents=parents)
  parser2.parse_args(rest, args)

if args.op == "a_axpx" and (args.matrixtype == "cqmat" or args.matrixtype == "cdiag") and args.NR != args.NC:
  print(f"The operation {args.op} needs a squarre matrix (with NR = NC).")
  sys.exit(1)

machine = importlib.import_module("machine." + args.machine)
header = machine.get_header(args)
ncores = machine.get_cores_per_node(args) * args.nodes

dict_to_pass = vars(args)
dict_to_pass["cores"] = ncores

command = f'python tools/run.py'
for k in ['resfile',  'machine', 'timeout']:
  command += f" --{k} {dict_to_pass[k]}"
command += ' " '

if args.lang == "MPIOMP":
  header += f"export OMP_NUM_THREADS={args.threads * args.tpc}\n"
  command += machine.get_mpirun(args) + " " + machine.get_mpirun_options_mpiomp(args) + " tbsla_perf_mpi_omp"

if args.lang == "OMP":
  header += f"export OMP_NUM_THREADS={args.threads * args.tpc}\n"
  command += " tbsla_perf_omp"

if args.lang == "MPI":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_mpi"

if args.lang == "PETSC":
  command += machine.get_mpirun(args) + f" -n {ncores} tbsla_perf_petsc"
  if args.format != "PETSC":
    print("The matrix format should be PETSC")
    sys.exit(1)

if args.lang == "HPX":
  if args.nodes == 1:
    command += f"tbsla_perf_hpx"
  else:
    command += machine.get_mpirun(args) + " " + machine.get_mpirun_options_hpx(args) + f" tbsla_perf_hpx -l {args.nodes}"

if args.lang == "YML":
  if args.LGC * args.LGR != args.CPT:
    print("LGC x LGR should be equal to CPT")
    sys.exit(1)
  if args.LGC * args.BGC != args.GC:
    print("LGC x BGC should be equal to GC")
    sys.exit(1)
  if args.LGR * args.BGR != args.GR:
    print("LGR x BGR should be equal to GR")
    sys.exit(1)
  header += "python tools/gen_yml_hostfile.py --hostfile ${HOME}/.omrpc_registry/nodes -n " + str(ncores + 1) + "\n"
  header += "python tools/reorder_machinefile.py --machinefile ${LSB_DJOB_HOSTFILE} -o _yml_tmpdir/machinefile_${LSB_BATCH_JID}\n\n"

  comp_dir_name = f'_yml_tmpdir/components/c{args.CPT}'
  if not os.path.isdir(comp_dir_name) and args.compilation == "True":
    cmd_compile_comp = machine.get_env(args)
    cmd_compile_comp += f"python src/yml/compilation/compile_components.py --C {args.CPT}\n"
    print(cmd_compile_comp)
    exe.execute_command(cmd_compile_comp)

  app_name = f'_yml_tmpdir/app/{args.op}_{args.CPT}_{args.BGR}_{args.BGC}.query'
  if (not os.path.isfile(app_name + ".yapp") or not os.path.isfile(app_name)) and args.compilation == "True":
    cmd_compile_app = machine.get_env(args)
    cmd_compile_app += f"python src/yml/compilation/compile_apps.py --C {args.CPT} --BGR {args.BGR} --BGC {args.BGC} --app {args.op}\n"
    print(cmd_compile_app)
    exe.execute_command(cmd_compile_app)

  if args.format == "COO":
    int_matrixformat = 1
  if args.format == "SCOO":
    int_matrixformat = 2
  if args.format == "CSR":
    int_matrixformat = 3
  if args.format == "ELL":
    int_matrixformat = 4
  if args.format == "DENSE":
    int_matrixformat = 5
  if args.matrixtype == "cqmat":
    for s in range(1, 2):
      for q in QLIST:
        dict_to_pass["Q"] = q
        dict_to_pass["S"] = s
        pack_name = f'_yml_tmpdir/param_nr{args.NR}_nc{args.NC}_{args.matrixtype}_c{args.C}_gr{args.GR}_gc{args.GC}_lgr{args.LGR}_lgc{args.LGC}.pack'
        command = 'rm -f ' + pack_name + '\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=n_row --integer={args.NR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=n_col --integer={args.NC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=C --integer={args.C}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=Q --integer={q}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=S --integer={s}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=gr --integer={args.GR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=gc --integer={args.GC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=lgr --integer={args.LGR}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=lgc --integer={args.LGC}\n'
        command += f'yml_parameter --app={app_name}.yapp  --pack={pack_name} --add=matrixformat --integer={int_matrixformat}\n'
        command += f'python tools/run_yml.py'
        for k in ['resfile',  'machine', 'timeout']:
          command += f" --{k} {dict_to_pass[k]}"
        command += ' " '
        command += machine.get_mpirun(args) + " -machine _yml_tmpdir/machinefile_${LSB_BATCH_JID} " + f"-n 1 yml_scheduler --input={pack_name}  {app_name}.yapp"
        sub_script = header + command + f'" --dic "{dict_to_pass}"\n\n'
        sub_script += machine.post_run_cmd(args) + "\n"
        sub_script += machine.post_processing(args) + "\n"
        fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_nr{args.NR}_nc{args.NC}_{args.matrixtype}_{args.format}_c{args.C}_gr{args.GR}_gc{args.GC}"
        fname += f"_cpt{args.CPT}_bgr{args.BGR}_bgr{args.BGC}_lgr{args.LGR}_lgc{args.LGC}_Q{q}_S{s}"
        fname += ".sh"
        sub.gen_submit_cmd(machine, args, fname, sub_script)
  else:
    command = f'python tools/run_yml.py'
    for k in ['resfile',  'machine', 'timeout']:
      command += f" --{k} {dict_to_pass[k]}"
    command += ' " '
    command += machine.get_mpirun(args) + f" -n 1 yml_scheduler"
    header += command +  f'" --dic "{dict_to_pass}"\n\n'
    header += machine.post_run_cmd(args) + "\n"
    header += machine.post_processing(args) + "\n"

    fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_nr{args.NR}_nc{args.NC}_{args.matrixtype}_{args.format}_c{args.C}_gr{args.GR}_gc{args.GC}"
    fname += f"_cpt{args.CPT}_bgr{args.BGR}_bgr{args.BGC}_lgr{args.LGR}_lgc{args.LGC}"
    fname += ".sh"
    sub.gen_submit_cmd(machine, args, fname, header)

if args.numainit == "True":
  command += " --numa-init"
if args.lang != "YML" and args.lang != "OMP":
  command += f" --GR {args.GR}"
  command += f" --GC {args.GC}"
if args.lang != "YML":
  command += f" --op {args.op}"
  command += f" --matrix {args.matrixtype}"
  if args.matrixfolder != ".":
    command += f" --matrix_folder {args.matrixfolder}"
  command += f" --format {args.format}"

  if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
    command += f" --NR {args.NR}"
    command += f" --NC {args.NC}"
    command += f" --C {args.C}"

  if args.matrixtype == "cqmat":
    incr = 1
    for s in range(1, 2):
      for q in QLIST:
        dict_to_pass["Q"] = q
        dict_to_pass["S"] = s
        header += command + f' --Q {q} --S {s}" --dic "{dict_to_pass}"'
        if hasattr(machine, 'OUT_DICT_FILE_CASES') and args.lang in machine.OUT_DICT_FILE_CASES and hasattr(machine, 'OUT_DICT_FILE') and machine.OUT_DICT_FILE != None:
          header += f' --infile ' + machine.OUT_DICT_FILE + f'.{incr}.0'
        header += '\n\n'
        incr += 1
  else:
    header += command +  f'" --dic "{dict_to_pass}"'
    if hasattr(machine, 'OUT_DICT_FILE_CASES') and args.lang in machine.OUT_DICT_FILE_CASES and hasattr(machine, 'OUT_DICT_FILE') and machine.OUT_DICT_FILE != None:
      header += f' --infile ' + machine.OUT_DICT_FILE + '.1.0'
    header += '\n\n'

  header += machine.post_processing(args) + "\n"
  fname = f"submit_{args.op}_{args.lang}_n{args.nodes}_{args.matrixtype}_{args.format}"
  if args.matrixtype == "cqmat" or args.matrixtype == "cdiag":
    fname += f"_nr{args.NR}_nc{args.NC}_c{args.C}"
  if args.lang != "OMP":
    fname += f"_gr{args.GR}_gc{args.GC}"
  if args.lang == "MPI" or args.lang == "HPX" or args.lang == "PETSC":
    fname += f".sh"
  elif args.lang == "MPIOMP" or args.lang == "OMP":
    fname += f"_t{args.threads}.sh"
  sub.gen_submit_cmd(machine, args, fname, header)
