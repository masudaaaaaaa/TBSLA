#/bin/bash
#PJM -N "make_build"
#PJM -L "node=2"
#PJM -L "rscgrp=small"
#PJM -L  "elapse=30:00"

set -e

export PROJECT_SOURCE_DIR=${HOME}/TBSLA/src
export INSTALL_DIR=${HOME}/install
export PATH=${INSTALL_DIR}/cmake/3.19.8/bin:$PATH

rm -rf ${INSTALL_DIR}/tbsla
#rm -rf _build
mkdir -p _build
cd _build

export OMP_NUM_THREADS=4

cmake -DTBSLA_ENABLE_VECTO=ON -DTBSLA_OpenMP_CXX_LIBRARIES="-Kopenmp" -DTBSLA_MPI_TESTS=ON -DCMAKE_CXX_FLAGS_RELEASE="-DNDEBUG" -DCMAKE_CXX_FLAGS="-Nclang -fPIC -Ofast -mcpu=native -funroll-loops -fno-builtin -march=armv8.2-a+sve" -DCMAKE_BUILD_TYPE=Release -DTBSLA_ENABLE_MPI=ON -DTBSLA_ENABLE_OMP=ON -DTBSLA_ENABLE_HPX=OFF -DMPI_CXX_COMPILER=mpiFCC -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR/tbsla ..
make -j VERBOSE=1
make test
make install
