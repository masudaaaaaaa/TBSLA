#ifndef TBSLA_MPI_MatrixCOO
#define TBSLA_MPI_MatrixCOO

#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixCOO : public tbsla::cpp::MatrixCOO, public virtual tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixCOO::spmv;
};

}}

#endif
