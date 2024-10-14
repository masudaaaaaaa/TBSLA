#ifndef TBSLA_CPP_MatrixDENSE
#define TBSLA_CPP_MatrixDENSE

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>

namespace tbsla { namespace cpp {

class MatrixDENSE : public virtual Matrix {
  public:
    MatrixDENSE() : values(NULL) {};
    MatrixDENSE(const tbsla::cpp::MatrixCOO & m);
    ~MatrixDENSE();
    friend std::ostream & operator<<( std::ostream &os, const MatrixDENSE &m);
    double* spmv(const double* v, int vect_incr = 0) const;
    inline void Ax(double* r, const double* v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    using tbsla::cpp::Matrix::AAxpAx;
    using tbsla::cpp::Matrix::AAxpAxpx;
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;
    void NUMAinit();
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);

  protected:
    double* values;
};

}}

#endif
