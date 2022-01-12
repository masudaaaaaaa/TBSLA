#OP=Ax_
#OP=AAxpAxpx
#OP=page_rank
#matrix=nlpkkt200
#matrix=nlpkkt120
#matrix=cage14
#matrix=cage15

#python tools/gen_submit_mm_cmd.py --OP $OP --Ns 1 --Ne 1 --machine Fugaku --matrixfolder ~/TBSLA/$matrix --matrixtype $matrix --numa-init --OMP
#python tools/gen_submit_mm_cmd.py --OP $OP --Ns 1 --Ne 2 --machine Fugaku --matrixfolder ~/TBSLA/$matrix --matrixtype $matrix --numa-init --MPI
#python tools/gen_submit_mm_cmd.py --OP $OP --Ns 1 --Ne 2 --machine Fugaku --matrixfolder ~/TBSLA/$matrix --matrixtype $matrix --numa-init --MPIOMP


#OP=Ax_
#N=8000000
#C=16
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 1 --machine Fugaku --numa-init --OMP
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 2 --machine Fugaku --numa-init --MPI
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 2 --machine Fugaku --numa-init --MPIOMP
#OP=AAxpAxpx
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 1 --machine Fugaku --numa-init --OMP
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 2 --machine Fugaku --numa-init --MPI
#python tools/gen_submit_cmd.py --C $C --OP $OP --NR $N --NC $N --Ns 1 --Ne 2 --machine Fugaku --numa-init --MPIOMP


OP=page_rank
N=1000000
NNZ=0.0001
C=0
python tools/gen_submit_cmd.py --C $C --NNZ $NNZ --OP $OP --NR $N --NC $N --Ns 1 --Ne 2 --machine Fugaku --numa-init --MPIOMP
