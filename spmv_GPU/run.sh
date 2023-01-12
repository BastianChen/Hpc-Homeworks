# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/gpfs/home/bxjs_01/spmv_data/spmv_data

EXECUTABLE=$1
REP=64

srun -p gpu ${EXECUTABLE} ${REP} ${DATAPATH}/circuit5M.csr
